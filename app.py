# Import TextLoader from LangChain.
# TextLoader reads a text file and converts it into a LangChain Document.
from langchain_community.document_loaders import TextLoader

# Import RecursiveCharacterTextSplitter.
# This is LangChain's most commonly used text splitter.
from langchain_text_splitters import RecursiveCharacterTextSplitter

#Step 1 : Document loading 
# Create a loader object and tell it which file to read.
loader = TextLoader("meeting.txt")

# Read the file.
# The result is a list of Document objects.
documents = loader.load()

#Step 2 : Chunking
splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,      # Maximum characters in one chunk
    chunk_overlap=20     # 20 characters overlap between chunks
)
# Split the document into chunks.
chunks = splitter.split_documents(documents)

# Print the number of chunks created.
print(f"Total Chunks: {len(chunks)}")

#enumerate gives you both index and a value. Basically it lets you loop through a list while keeping the track of index.
# Print every chunk.
for i, chunk in enumerate(chunks):
    print("=" * 40)
    print(f"Chunk {i+1}")
    print("=" * 40)
    print(chunk.page_content)
    
#page_content is an attribute (property) of a LangChain Document object that stores the actual text of the document or chunk.

#chunk_size is the maximum size of a chunk, not the exact size. RecursiveCharacterTextSplitter tries to keep meaningful text together and only splits when necessary, while chunk_overlap repeats a small part of the previous chunk to preserve context.