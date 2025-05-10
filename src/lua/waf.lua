-- waf_api.lua
-- 基础 WAF API，实现对请求进行规则判断和处理

local _M = {}
local cjson = require("cjson.safe")
local ngx_re = require("ngx.re")
local str = require("resty.string")
local base64 = require("ngx.base64")
local http = require "resty.http"
-- 共享字典引用
local ipBlock = ngx.shared.ip_block
local ipCache = ngx.shared.ip_cache

-- 通用工具函数
function _M.startWith(sstr, dstr)
    return sstr:sub(1, #dstr) == dstr
end

function _M.endWith(sstr, dstr)
    return sstr:sub(-#dstr) == dstr
end

function _M.toLower(sstr)
    return sstr:lower()
end

function _M.contains(sstr, dstr)
    return sstr:find(dstr, 1, true) ~= nil
end

function _M.regex(sstr, pat, ext)
    return ngx.re.match(sstr, pat, ext)
end

function _M.rgxMatch(sstr, pat, ext)
    local m, _ = ngx.re.match(sstr, pat, ext)
    return m ~= nil
end

function _M.rgxGmatch(sstr, pat, ext)
    return ngx.re.gmatch(sstr, pat, ext)
end

function _M.rgxSub(subject, regex, replace, options)
    return ngx.re.sub(subject, regex, replace, options)
end

function _M.rgxGsub(subject, regex, replace, options)
    return ngx.re.gsub(subject, regex, replace, options)
end

function _M.kvFilter(v, match, valOnly)
    for k, val in pairs(v or {}) do
        if valOnly then
            if match(val) then return true, val end
        else
            if match(k) or match(val) then return true, k end
        end
    end
    return false, nil
end

function _M.knFilter(v, match, p)
    for i, name in ipairs(v.name or {}) do
        local content = v.name and v.name[i]
        local test = p == 1 and name or content
        if test and match(test) then return true, test end
    end
    return false, nil
end

function _M.jsonFilter(v, match, parsed, valOnly)
    local ok, obj = pcall(function()
        return parsed and v or cjson.decode(v)
    end)
    if not ok or type(obj) ~= "table" then return false, nil end
    for k, val in pairs(obj) do
        if valOnly then
            if match(val) then return true, val end
        else
            if match(k) or match(val) then return true, k end
        end
    end
    return false, nil
end

function _M.base64Decode(str)
    local ok, decoded = pcall(ngx.decode_base64, str)
    if ok then return decoded else return nil end
end

function _M.strCounter(sstr, dstr)
    local count = 0
    for _ in sstr:gmatch(dstr) do
        count = count + 1
    end
    return count
end

function _M.trim(str)
    return (str:gsub("^%s*(.-)%s*$", "%1"))
end

function _M.inArray(str, arr)
    for _, v in ipairs(arr) do
        if v == str then return true end
    end
    return false
end

function _M.pmMatch(sstr, dict)
    for _, pattern in ipairs(dict) do
        if sstr:find(pattern, 1, true) then
            return true, pattern
        end
    end
    return false, nil
end

function _M.urlDecode(sstr)
    return ngx.unescape_uri(sstr)
end

function _M.htmlEntityDecode(sstr)
    return (sstr:gsub("&lt;", "<"):gsub("&gt;", ">"):gsub("&amp;", "&"):gsub("&quot;", '"'):gsub("&#39;", "'"))
end

function _M.hexDecode(sstr)
    return sstr:gsub("%%(%x%x)", function(h)
        return string.char(tonumber(h, 16))
    end)
end

function _M.block(reset)
    if reset then
        ngx.exit(444)
    else
        ngx.status = 403
        ngx.say("Access Denied")
        ngx.exit(403)
    end
end

function _M.redirect(uri, status)
    ngx.redirect(uri, status or 302)
end

function _M.errLog(...)
    ngx.log(ngx.ERR, ...)
end

-- 语义引擎检测（占位，需对接模型）
function _M.checkSQLI(str, level) return false end
function _M.checkRCE(str, level) return false end
function _M.checkPT(str) return false end
function _M.checkXSS(str) return false end
function _M.checkRobot(waf) return false end
function _M.ip2loc(ip, lang) return "中国", "湖北省", "武汉市" end


-- 获取客户端真实 IP
function _M.get_client_ip()
    local headers = ngx.req.get_headers()
    local ip = headers["X-Real-IP"] or headers["X-Forwarded-For"] or ngx.var.remote_addr
    if ip then
        -- 如果是多个 IP（如 X-Forwarded-For），取第一个
        ip = ip:match("([^,]+)")
    end
    return ip
end

function _M.get_cookies()
    local cookies = {}
    local cookie_header = ngx.var.http_cookie
    if cookie_header then
        for k, v in string.gmatch(cookie_header, "([^=; ]+)=([^;]+)") do
            cookies[k] = v
        end
    end
    return cookies
end

-- 初始化 waf 表（每次请求执行）
function _M.build_waf_context()
    local headers = ngx.req.get_headers()
    local uri_args = ngx.req.get_uri_args()
    ngx.req.read_body()
    local post_args = ngx.req.get_post_args()

    local waf = {
        ip = _M.get_client_ip(),
        scheme = ngx.var.scheme,
        httpVersion = tonumber(ngx.req.http_version()),
        host = headers["Host"],
        ipBlock = ipBlock,
        ipCache = ipCache,
        requestLine = ngx.var.request,
        uri = ngx.var.uri,
        method = ngx.req.get_method(),
        reqUri = ngx.var.request_uri,
        userAgent = headers["User-Agent"],
        referer = headers["Referer"],
        reqContentType = headers["Content-Type"],
        XFF = headers["X-Forwarded-For"],
        origin = headers["Origin"],
        reqHeaders = headers,
        hErr = nil,
        isQueryString = next(uri_args) ~= nil,
        reqContentLength = tonumber(headers["Content-Length"] or 0),
        queryString = uri_args,
        qErr = nil,
        form = {
            RAW = ngx.req.get_body_data(),
            FORM = post_args,
            FILES = {},
        },
        fErr = nil,
        cookies = _M.get_cookies(),
        cErr = nil,
        status = nil,
        respHeaders = {},
        respContentLength = nil,
        respContentType = nil,
        respBody = nil,
        replaceFilter = false,
    }
    return waf
end

-- 默认禁止函数
function _M.forbidden(reason)
    ngx.status = ngx.HTTP_FORBIDDEN
    ngx.header["Content-Type"] = "application/json"
    ngx.say(cjson.encode({ code = 510, msg = reason or "Forbidden" }))
    return ngx.exit(ngx.HTTP_FORBIDDEN)
end

local function build_input_data(waf)
    local input_data = {}

    -- 收集请求参数
    table.insert(input_data, waf.reqUri)
    table.insert(input_data, waf.userAgent or "")
    table.insert(input_data, waf.method)
    table.insert(input_data, waf.form.RAW or "")
    -- 可以根据需要增加其他特征
    table.insert(input_data, waf.referer or "")
    table.insert(input_data, waf.origin or "")
    -- 删除掉空值
    for i = #input_data, 1, -1 do
        if input_data[i] == nil or input_data[i] == "" then
            table.remove(input_data, i)
        end
    end
    return input_data
end

local ONNX_SERVICE_URL = os.getenv("ONNX_SERVICE_URL") or "http://127.0.0.1:5555/detect"

function _M.check_with_model(waf)
    local input_data = {texts = {}, max_len = 100}
    input_data.texts = build_input_data(waf)
    ngx.log(ngx.INFO, "Input data for model: ", cjson.encode(input_data))
    local httpc = http.new()
    ngx.log(ngx.INFO, "Model API Host: ", ONNX_SERVICE_URL)
    local res, err = httpc:request_uri(ONNX_SERVICE_URL, {
        method = "POST",
        body = cjson.encode(input_data),
        headers = {
            ["Content-Type"] = "application/json",
        },
        keepalive = false
    })

    if not res then
        ngx.log(ngx.ERR, "Model request failed: ", err)
        return
    else
        ngx.log(ngx.INFO, "Model response: ", res.body)
    end

    local result = cjson.decode(res.body)
    if result.max_score > 0.8 then
        return _M.forbidden("Blocked by semantic model: " .. (result.text or "unknown"))
    end
end

-- 示例规则执行器
function _M.run_rules(waf)
    if waf.ipBlock:get(waf.ip) then
        return _M.forbidden("IP in blacklist")
    end

    -- 调用模型进行语义检测
    local model_check = _M.check_with_model(waf)
    if model_check then
        return model_check
    end

    -- 示例规则检查
    if waf.method == "GET" and waf.reqUri and waf.reqUri:lower():find("select.+from") then
        return _M.forbidden("SQLi detected")
    end

    -- local ua = waf.userAgent or ""
    -- if ua:find("curl") or ua:find("sqlmap") then
    --     return _M.forbidden("Bad UA")
    -- end

    -- if waf.form.RAW and waf.form.RAW:lower():find("union.*select") then
    --     return _M.forbidden("Suspicious POST data")
    -- end
end

return _M
