from langchain_core.documents import Document

#creating a sample document
def smaple_doc():
    sample_doc = Document(
        page_content="This is my sample document",
        metadata = {
            "a":"1"
        }
    )

    print(sample_doc)
    #page_content='This is my sample document' metadata={'a': '1'}

smaple_doc()