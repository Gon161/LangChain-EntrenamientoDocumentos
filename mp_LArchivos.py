
import io
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import OllamaEmbeddings
from dotenv import load_dotenv
from langchain.vectorstores.pgvector import PGVector
import configparser
from openai import OpenAI

import fitz
from langchain.chains import RetrievalQA
from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.document import Document
from langchain_community.llms import Ollama
import base64
from langchain.text_splitter import TokenTextSplitter
from langchain.chains import MapReduceDocumentsChain
from langchain.chains import ReduceDocumentsChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain


def ProcesarSolicitud(sMetodo,sParametros):
    resultado_del_metodo=""
    config = configparser.ConfigParser()
    config.read('config.ini', encoding='utf-8')
    COLLECTION_NAME = ""
    CONNECTION_STRING = config['AppSettings']['CONNECTION_STRING']

    match (sMetodo):
        case "PDF":
            sNombreDocumento, pdf_codificado, sNombrePDF = sParametros.split('|')
            pdf_decodificado = base64.b64decode(pdf_codificado)
            COLLECTION_NAME = sNombreDocumento
            pdf_stream = io.BytesIO(pdf_decodificado)
            doc = fitz.open(stream=pdf_stream, filetype="pdf")

            ggTXTCompleto = ""

            # Iterate through each page and extract text
            for page_num in range(doc.page_count):
                page = doc[page_num]
                ggTXTCompleto += page.get_text()
                images = page.get_images(full=True)
                tabs = page.find_tables()
                
                for table_index, tab in enumerate(tabs):
                    df = tab.to_pandas()
                    ggTXTCompleto += "\n\n"
                    
                    column_headers = [str(col).replace('\n', ' ') for col in df.columns]
                    ggTXTCompleto += "~".join(column_headers) + "\n"

                    for index, row in df.iterrows():
                        processed_row = [str(cell).replace('\n', ' ') for cell in row]
                        ggTXTCompleto += "~".join(processed_row) + "\n"
                        ggTXTCompleto += ""
                ggTXTCompleto += "\n\n"

            doc.close()

            text_splitter = CharacterTextSplitter()
            texts = text_splitter.split_text(ggTXTCompleto)
            docs = [Document(page_content=t) for t in texts]

            ###########################################################################
            ########################## Seccion Resumen ################################
            ###########################################################################
            
            map_template = """Escribe un resumen en español del siguiente documento:

            {content}

            """
            map_prompt = PromptTemplate.from_template(map_template)

            llm = Ollama(model="llama2")
            map_chain = LLMChain(prompt=map_prompt, llm=llm)
            reduce_template = """El siguiente es un conjunto de resumenes en español:

            {doc_summaries}

            Realiza un resumen con la infotmacion dada de los resumenes 
            """
            reduce_prompt = PromptTemplate.from_template(reduce_template)
            reduce_chain = LLMChain(prompt=reduce_prompt, llm=llm)
            stuff_chain = StuffDocumentsChain(
                llm_chain=reduce_chain, document_variable_name="doc_summaries")

            reduce_chain = ReduceDocumentsChain(
                combine_documents_chain=stuff_chain,
            )
            map_reduce_chain = MapReduceDocumentsChain(
                llm_chain=map_chain,
                document_variable_name="content",
                reduce_documents_chain=reduce_chain
            )
            splitter = TokenTextSplitter(chunk_size=2000)
            split_docs = splitter.split_documents(docs)
            ResumenFinal = map_reduce_chain.run(split_docs)
            
            ###########################################################################
            ######################### Seccion WordCloud ###############################
            ###########################################################################
            
            map_template = """Realiza un lista del siguiente text con las palabras clave:

            {content}

            """
            map_prompt = PromptTemplate.from_template(map_template)
            llm = Ollama(model="llama2")
            map_chain = LLMChain(prompt=map_prompt, llm=llm)
            reduce_template = """Obtene todas las palabras clave con del siguiente contenido y enlistalas en español:

            {doc_summaries}
            
            solo realiza un listado
            """
            reduce_prompt = PromptTemplate.from_template(reduce_template)
            reduce_chain = LLMChain(prompt=reduce_prompt, llm=llm)
            stuff_chain = StuffDocumentsChain(
                llm_chain=reduce_chain, document_variable_name="doc_summaries")

            reduce_chain = ReduceDocumentsChain(
                combine_documents_chain=stuff_chain,
            )
            map_reduce_chain = MapReduceDocumentsChain(
                llm_chain=map_chain,
                document_variable_name="content",
                reduce_documents_chain=reduce_chain
            )
            splitter = TokenTextSplitter(chunk_size=2000)
            split_docs = splitter.split_documents(docs)
            PalabrasClave = map_reduce_chain.run(split_docs)
            
            load_dotenv()
            
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20,length_function=len,is_separator_regex=False,)
            texts = text_splitter.split_documents(docs)
            embeddings = OllamaEmbeddings(model="llama2")
            doc_vectors = embeddings.embed_documents([t.page_content for t in texts[:5]])
            db = PGVector.from_documents(
                embedding=embeddings,
                documents=texts,
                collection_name=COLLECTION_NAME,
                connection_string=CONNECTION_STRING
            )

            return "Proceso Terminado"
        
        case "XML":

            return "Proximamente"

        case "Excel":
            return "Proximamente"
            
        case "Image":
            return "Proximamente"

        case "Audio":
            return "Proximamente"
            
        case "Video":
            return "Proximamente"

        case "PreguntaDocumento":
            pregunta, COLLECTION_NAME= sParametros.split('|')
            embeddings = OpenAIEmbeddings()
            db = PGVector.from_documents(
                    embedding=embeddings,
                    documents="",
                    collection_name=COLLECTION_NAME,
                    connection_string=CONNECTION_STRING
                )
            
            
            # similar = db.similarity_search_with_score(query, k=3)
            retriever = db.as_retriever(search_kwargs={'k': 3})  # default 4
   
            qa_chain = RetrievalQA.from_chain_type(
                llm=OpenAI(),chain_type="stuff", retriever=retriever, return_source_documents=True
            )
            result = qa_chain({"query": pregunta})
            return result["result"]