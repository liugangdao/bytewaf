import random
import csv


def _generate_payloads(templates, count=100):
    """
    生成结构性正常的 Payload 数据集
    :param templates: 模板列表
    :param count: 生成数量
    :return: 生成的 Payload 列表
    """
    # SQL 和 XSS 常见关键词
    sql_keywords = [
        "select",
        "insert",
        "update",
        "delete",
        "drop",
        "table",
        "from",
        "where",
        "having",
        "order by",
        "group by",
        "union",
    ]
    xss_keywords = ["<script>", "alert", "onerror", "onload", "eval", "document.cookie"]

    # Payload 模板构建（结构保留符号 = ; "" () <>）
    sql_templates = [
        "select={field}&from={table}",
        "update={table}&set=field='{value}'",
        "delete=0;drop={table};",
        "table={table}&order_by={column}",
        "group_by={column}&having=count>{n}",
        "union=all&select={field}",
    ]

    xss_templates = [
        "<script>alert='{msg}';</script>",
        "onerror=\"logError('{code}')\"",
        "onload=\"init('{msg}')\"",
        'document.cookie="user={id}; path=/"',
        "eval(\"load('{param}')\")",
    ]

    # 参数替代词库
    fields = ["name", "age", "email", "status"]
    tables = ["users", "orders", "logs", "settings"]
    columns = ["created_at", "type", "level"]
    values = ["active", "yes", "123"]
    msgs = ["hello", "warning", "limit reached"]
    codes = ["404", "500", "403"]
    ids = ["abc123", "guest", "token_456"]
    params = ["config", "theme"]

    # 填充模板
    def fill_template(template):
        return template.format(
            field=random.choice(fields),
            table=random.choice(tables),
            value=random.choice(values),
            column=random.choice(columns),
            msg=random.choice(msgs),
            code=random.choice(codes),
            id=random.choice(ids),
            param=random.choice(params),
            n=random.randint(1, 10),
        )

    # 生成 payload
    def generate_payloads(templates, count=100):
        return [
            (fill_template(t), "non_attack") for t in random.choices(templates, k=count)
        ]

    # 生成数据
    sql_payloads = generate_payloads(sql_templates, 100)
    xss_payloads = generate_payloads(xss_templates, 100)

    # 写入文件
    with open("structured_normal_payloads.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["payload", "label"])
        writer.writerows(sql_payloads + xss_payloads)

    print("✅ 结构性正常 Payload 数据集已生成：structured_normal_payloads.csv")
    payloads = []
    for _ in range(count):
        template = random.choice(templates)
        payload = template.format(
            field=random.choice(fields),
            table=random.choice(tables),
            value=random.choice(values),
            column=random.choice(columns),
            msg=random.choice(msgs),
            code=random.choice(codes),
            id=random.choice(ids),
            param=random.choice(params),
            n=random.randint(1, 10),
        )
        payloads.append((payload, "non_attack"))
    return payloads


def _gen_normal():
    import csv

    # SQL 关键词扩展
    sql_keywords = [
        "user",
        "password",
        "select",
        "insert",
        "update",
        "delete",
        "drop",
        "truncate",
        "replace",
        "create",
        "alter",
        "rename",
        "table",
        "column",
        "from",
        "where",
        "having",
        "order",
        "by",
        "group",
        "into",
        "values",
        "set",
        "and",
        "or",
        "not",
        "null",
        "is",
        "like",
        "between",
        "in",
        "exists",
        "union",
        "all",
        "as",
        "join",
        "on",
        "case",
        "when",
        "then",
        "else",
        "if",
        "procedure",
        "execute",
        "limit",
        "offset",
        "cast",
        "convert",
        "declare",
        "fetch",
        "cursor",
    ]

    # XSS 关键词扩展
    xss_keywords = [
        "script",
        "alert",
        "confirm",
        "prompt",
        "Function",
        "setTimeout",
        "setInterval",
        "document",
        "window",
        "location",
        "document.cookie",
        "document.write",
        "innerHTML",
        "outerHTML",
        "onload",
        "onerror",
        "onmouseover",
        "onfocus",
        "onblur",
        "onchange",
        "onclick",
        "onkeydown",
        "onkeyup",
        "onkeypress",
        "onmouseenter",
        "onmouseleave",
        "src",
        "data",
        "href",
        "iframe",
        "img",
        "svg",
        "math",
        "style",
        "expression",
    ]

    # 合并并去重
    all_keywords = sorted(set(sql_keywords + xss_keywords))
    all_keywords_eq = [s + "=" for s in all_keywords]
    all_keywords_plus = [s + "+" for s in all_keywords]
    all_keywords_and = [s + "&" for s in all_keywords]
    all_keywords = all_keywords + all_keywords_eq + all_keywords_plus + all_keywords_and
    # 写入 CSV
    with open(
        "data/rich_keywords_non_attack.csv", "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.writer(f)
        writer.writerow(["sample", "label"])
        for keyword in all_keywords:
            writer.writerow([keyword, 0])

    print("✅ 扩展关键词数据集已生成：rich_keywords_non_attack.csv")


if __name__ == "__main__":
    _gen_normal()
