from typing import Union
from bs4 import BeautifulSoup, Tag
import pandas as pd


def clean_text(text: str):
    text = " ".join(text.split())
    return text


class DocumentParser:
    """
    Semantic HTML parser for EURLEX documents. This parser separates the document into different sections and
    extracts the contents together with some metadata like titles and tables from the document.
    """

    def __init__(self, soup: BeautifulSoup, link: str):
        self.doc = link.split("uri=")[-1].replace(":", "_")
        self.data = pd.DataFrame(columns=["doc", "index", "type", "content"])
        self.soup = soup
        self.metadata = self._get_metadata(soup)
        self.metadata["link"] = link
        self.idx = 0
        self._parse_doc(soup)

    def _add_content(self, content, type):
        content = clean_text(content)
        self.data.loc[self.idx] = [self.doc, self.idx, type, content.strip()]
        self.idx += 1

    def _parse_doc(self, soup: Union[BeautifulSoup, Tag]):
        if soup.__class__ == BeautifulSoup and soup.body is not None:
            content = soup.body.contents
        else:
            content = soup.contents
        for cont in content:
            if cont.name == "table":
                self._parse_table(cont)
            elif cont.name == "div":
                self._parse_doc(cont)
            elif cont.name == "p" or cont.name == "span":
                self._parse_paragraph(cont)
            elif cont.name == "h1":
                self._add_content(cont.text, "header")
            elif cont.name == "img" or cont.name == "figure":
                # Image handling?
                continue
            elif (
                cont.name == "script"
                or cont.name is None
                or cont.name == "hr"
                or cont.name == "br"
                or cont.name == "em"
                or cont.name == "a"
            ):
                # not important
                continue
            else:
                print("Unknown element: ", cont.name)
                print(cont)
                print(self.metadata["link"])

    def _parse_paragraph(self, paragraph: Tag):
        class_name = paragraph.get("class")
        type_name = "p"
        if class_name is not None:
            type_name += "-" + class_name[0]
        match type_name:
            case "p-doc-ti" | "p-oj-doc-ti":
                if str(paragraph.text).startswith("of "):
                    date = self._parse_date(paragraph)
                    if date is not None:
                        self.metadata["date"] = date
                else:
                    self._add_content(paragraph.text, type="title")
            case "p-image" | "p-oj-separator" | "p-separator" | "p-bglang":
                return
            case "p-normal" | "p-oj-normal":
                self._add_content(paragraph.text, type="text")
            case "p-note" | "p-oj-note":
                self._add_content(paragraph.text, type="footnote")  # save href?
            case "p-oj-no-doc-c" | "p-no-doc-c" | "p-oj-sti-art" | "p-sti-art":
                self._add_content(paragraph.text, type="subtitle")
            case "p-oj-signatory" | "p-signatory":
                self._add_content(paragraph.text, type="signatory")
            case "p-oj-ti-annotation" | "p-ti-annotation":
                self._add_content(paragraph.text, type="annotation")
            case "p-oj-ti-art" | "p-ti-art":
                self._add_content(paragraph.text, type="article_no")
            case "p-oj-ti-grseq-1" | "p-ti-grseq-1":
                self._add_content(paragraph.text, type="subtitle")  # subsubtitle?
            case "p-oj-ti-section-1" | "p-ti-section-1":
                self._add_content(paragraph.text, type="subsection_no")
            case "p-oj-ti-section-2" | "p-ti-section-2":
                self._add_content(paragraph.text, type="subtitle")
            case "p-oj-ti-tbl" | "p-ti-tbl":
                self._add_content(paragraph.text, type="table_title")
            case "p":
                footnote = paragraph.find(name="a")
                if footnote is not None:
                    print(footnote)
                self._add_content(paragraph.text, type="text")
            case _:
                print("Unknown paragraph type: ", type_name)

    def _parse_date(self, paragraph: Tag):
        elements = str(paragraph.text).split(" ")[1:]
        if len(elements) != 3:
            return
        month = ""
        match elements[1]:
            case "January":
                month = "01"
            case "February":
                month = "02"
            case "March":
                month = "03"
            case "April":
                month = "04"
            case "May":
                month = "05"
            case "June":
                month = "06"
            case "July":
                month = "07"
            case "August":
                month = "08"
            case "September":
                month = "09"
            case "October":
                month = "10"
            case "November":
                month = "11"
            case "December":
                month = "12"
        if len(elements[0]) == 1:
            elements[0] = "0" + elements[0]
        if month == "":
            return
        date = elements[0] + "." + month + "." + elements[2]
        return date

    def _parse_table(self, table: Tag):
        contents = table.contents
        for element in contents:
            if element.name == "col" or element.name == None:
                continue
            if element.name == "tbody":
                self._parse_table(element)
            elif element.name == "tr":
                self._parse_table_row(element)
            else:
                print("Unknown table element: ", element.name)

    def _parse_table_row(self, row: Tag):
        contents = row.contents
        row_content = ""
        for element in contents:
            if element.name is None:
                continue
            if element.name == "td":
                row_content += " " + element.text.strip()
            else:
                print("Unknown row element: ", element.name)
        self._add_content(row_content.strip().replace("\n", " "), "table_row")

    def _parse_header(self, soup: BeautifulSoup, metadata: dict):
        if soup.body is None:
            return
        header = soup.body.find("table")
        if header is None:
            return
        date = header.find(class_="hd-date")
        oj = header.find(class_="hd-oj")
        if date is not None:
            metadata["date_published"] = date.text.strip()
        if oj is not None:
            metadata["oj"] = oj.text
        lang = header.find(class_="hd-lg")
        if lang is not None:
            metadata["language"] = lang.text

    def _get_metadata(self, soup: BeautifulSoup) -> dict:
        metadata = {}
        for meta in soup.findAll("meta"):
            name = meta.get("name")
            content = meta.get("content")
            if name == None or content == None:
                continue
            if "language" in name:
                metadata["language"] = content
            elif "title" in name:
                metadata["doctitle"] = content
            elif "subject" in name:
                metadata["subject"] = content
            elif "description" in name:
                metadata["description"] = content
            elif "source" in name:
                metadata["source"] = content
            elif "publisher" in name:
                metadata["publisher"] = content
        self._parse_header(soup, metadata)
        return metadata
