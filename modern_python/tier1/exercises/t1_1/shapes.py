"""T1-1 练习 2 起始代码：用 Protocol 解耦。

现状：render_report 硬依赖 PdfExporter。想支持 HtmlExporter，
Java 里你会抽 interface 再让两个类 implements；Python 里请用 Protocol。
"""


class PdfExporter:
    def export(self, title, rows):
        lines = [f"# {title}"]
        for name, score in rows:
            lines.append(f"- {name}: {score}")
        return "\n".join(lines)


def render_report(exporter, title, rows):
    return exporter.export(title, rows)


if __name__ == "__main__":
    rows = [("alice", 90), ("bob", 85)]
    print(render_report(PdfExporter(), "成绩单", rows))
