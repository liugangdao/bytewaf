FROM openresty/openresty:1.27.1.2-0-alpine-fat

# 安装 luarocks
RUN apk add --no-cache git curl build-base luarocks

# 安装 lua-resty-http
RUN luarocks install lua-resty-http

# 拷贝配置和 Lua 脚本
COPY nginx.conf /usr/local/openresty/nginx/conf/nginx.conf
COPY lua/waf.lua /usr/local/openresty/nginx/lua/waf.lua

# 默认启动脚本
CMD ["/usr/local/openresty/bin/openresty", "-g", "daemon off;"]
