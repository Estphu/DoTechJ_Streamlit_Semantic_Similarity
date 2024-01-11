import streamlit as st # Streamlit Framework Python Pkg
import PyPDF2 # Pkg for PDF Reader
# from docx import Document # Pkg for Docx Reader
# import docx2txt # Pkg for conversion docx to txt
from sentence_transformers import SentenceTransformer # Pkg for converting the extracted texts into vector embeddings
from annoy import AnnoyIndex # Pkg for storing indexing and embeddings
from scipy.spatial.distance import cosine # Pkg for calculating semantic similarity between embeddings

# Function to extract text from a PDF document
def extract_text_pdf(file):
    try:
        pdf_reader = PyPDF2.PdfReader(file)
        text = ""
        for page_num in range(len(pdf_reader.pages)):
            text += pdf_reader.pages[page_num].extract_text()
        return text
    except PyPDF2.utils.PdfReadError:
        st.error(f"Error extracting text from PDF: {file.name}")
        return ""

# Function to extract text from a DOC document
# def read_docx(file_path):
#     doc = Document(file_path)
#     text = ""
#     for paragraph in doc.paragraphs:
#         text += paragraph.text + "\n"
#     return text

# def read_doc(file_path):
#     text = docx2txt.process(file_path)
#     return text
    
# Function to embed sentence into vector reresentation
def embed_text(text, model):
    return model.encode(text, convert_to_tensor=True)

# Storing embedding through db
def index_embeddings(embeddings):
    dim = len(embeddings[0])
    index = AnnoyIndex(dim, 'euclidean')
    for i, vec in enumerate(embeddings):
        index.add_item(i, vec)
    index.build(10)  # Number of trees for the index
    return index

# Calculating similarity between two embedding using scipy
def calculate_similarity(embedding1, embedding2):
    return 1 - cosine(embedding1, embedding2)

def main():
    # Title
    st.title("Semantic Similarity Assessment")

    # Loading pre-trained sentence embeddings model
    model_name = "paraphrase-MiniLM-L6-v2"
    model = SentenceTransformer(model_name)

    # Upload documents
    uploaded_file1 = st.file_uploader("Upload Document 1", type=["pdf", "doc", "docx"])
    uploaded_file2 = st.file_uploader("Upload Document 2", type=["pdf", "doc", "docx"])

    print(uploaded_file1)

    if uploaded_file1 is not None and uploaded_file2 is not None:
        # Extracting text from documents
        if uploaded_file1.type == 'application/pdf':
            text1 = extract_text_pdf(uploaded_file1)
        # elif uploaded_file1.type in ['application/msword', 'application/vnd.openxmlformats-officedocument.wordprocessingml.document']:
        #     text1 = extract_text_doc(uploaded_file1)
        else:
            st.error("Unsupported file format. Please upload PDF or DOC files.")
        st.text_area("Content of Document 1", text1)
        embedding_doc1 = embed_text(text1, model)

        if uploaded_file2.type == 'application/pdf':
            text2 = extract_text_pdf(uploaded_file2)
        # elif uploaded_file2.type in ['application/msword', 'application/vnd.openxmlformats-officedocument.wordprocessingml.document']:
        #     text2 = extract_text_doc(uploaded_file2)
        else:
            st.error("Unsupported file format. Please upload PDF or DOC files.")
        st.text_area("Content of Document 2", text2)
        embedding_doc2 = embed_text(text2, model)

        # Index and store embeddings
        all_embeddings = [embedding_doc1.cpu().numpy(), embedding_doc2.cpu().numpy()]
        index = index_embeddings(all_embeddings)

        # Calculating semantic similarity
        similarity_score = calculate_similarity(embedding_doc1.cpu().numpy(), embedding_doc2.cpu().numpy())
        
        # Converting similarity score to percentage
        similarity_percentage = similarity_score * 100

        st.subheader("Semantic Similarity:")
        st.write(f"The semantic similarity score between the two documents is: {similarity_percentage:.2f}%")

if __name__ == '__main__':
    main()
