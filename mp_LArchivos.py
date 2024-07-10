""""
Documentacion de pgvector 
https://github.com/pgvector/pgvector

Videos de referencia 
https://www.youtube.com/@saxsabigdata1400


"""


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
    #Cadena de conexion a postgresql con el plugin pgvector 
    CONNECTION_STRING = config['AppSettings']['CONNECTION_STRING']

    match (sMetodo):
        case "PDF":
            # Nombre de la empres, pdf en base 64, Nombre del PDF
            sNombreEmpresa, pdf_codificado, sNombrePDF = sParametros.split('|')
            # decodificar el pdf
            pdf_decodificado = base64.b64decode(pdf_codificado)
            # identificador del documento 
            COLLECTION_NAME = sNombreEmpresa
            #cargar el pdf en memoria para su uso
            pdf_stream = io.BytesIO(pdf_decodificado)
            doc = fitz.open(stream=pdf_stream, filetype="pdf")
            #variable que almacena del texto del documento 
            ggTXTCompleto = ""

            # ciclo que pasa por cada hoja 
            for page_num in range(doc.page_count):
                #obtener la pagina a utilizar 
                page = doc[page_num]
                #obtener todo el texto de la pagina 
                ggTXTCompleto += page.get_text()
                #obtener todas la imagenes 
                images = page.get_images(full=True)
                #obtener todas las tablas
                tabs = page.find_tables()
                #recorrer las tablas encontradas
                for table_index, tab in enumerate(tabs):
                    #pasar la tabla a una tabla de pandas 
                    df = tab.to_pandas()
                    #generar salto de linea
                    ggTXTCompleto += "\n\n"
                    #obtener el cabezado de cada tabla 
                    column_headers = [str(col).replace('\n', ' ') for col in df.columns]
                    #generar una separacion con el simbolo ~
                    ggTXTCompleto += "~".join(column_headers) + "\n"
                    #recorrer cada fila de la tabla y realizar su seperacion de ~ 
                    for index, row in df.iterrows():
                        processed_row = [str(cell).replace('\n', ' ') for cell in row]
                        ggTXTCompleto += "~".join(processed_row) + "\n"
                        ggTXTCompleto += ""
                ggTXTCompleto += "\n\n"
            #cerrar documento 
            doc.close()
            #se pasa el texto de nuevo a un documento para se procesado por el plugin 
            text_splitter = CharacterTextSplitter()
            texts = text_splitter.split_text(ggTXTCompleto)
            docs = [Document(page_content=t) for t in texts]

            load_dotenv()
            #realizar el separado de parrafos 
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20,length_function=len,is_separator_regex=False,)
            texts = text_splitter.split_documents(docs)
            #inicializar el modelo para el procesamiento de lenguaje natural
            embeddings = OllamaEmbeddings(model="llama3")
            #vectorizar los parrafos para guardar la informacion en la base de datos
            doc_vectors = embeddings.embed_documents([t.page_content for t in texts[:5]])
            #registrar los vectores
            db = PGVector.from_documents(
                embedding=embeddings,
                documents=texts,
                collection_name=COLLECTION_NAME,
                connection_string=CONNECTION_STRING
            )
            return "Proceso Terminado"
        case "PreguntaDocumento":
            #Pregunta, Identificador del documento
            pregunta, COLLECTION_NAME= sParametros.split('|')
            #iniciar el modelo para el procesamiento del lenguaje natural 
            embeddings =OllamaEmbeddings(model="llama3")
            #realizar la conexion a la base de datos vectorial 
            db = PGVector.from_documents(
                    embedding=embeddings,
                    documents="",
                    collection_name=COLLECTION_NAME,
                    connection_string=CONNECTION_STRING
                )
            # configurar el retriever, extraer las 3 vectores mas similares a la pregunta 
            retriever = db.as_retriever(search_kwargs={'k': 3})  # default 4
            # inicializar el RetrievalQA
            qa_chain = RetrievalQA.from_chain_type(
                llm=Ollama("llama3"),chain_type="stuff", retriever=retriever, return_source_documents=True
            )
            #generar la respuesta procesada por el modelo con los vectores cercanos a la pregunta 
            result = qa_chain({"query": pregunta})
            #retornar la respuesta 
            return result["result"]
