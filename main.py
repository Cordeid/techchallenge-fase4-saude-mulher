from scripts import step06_pdf
from scripts.step01_ingestao import processar_casos as step01_ingestao
from scripts.step02_audio import processar_caso as step02_audio
from scripts.step03_video import processar_caso as step03_video
from scripts.step04_rag import processar_caso as step04_rag, inicializar_rag
from scripts.step05_fusao import processar_caso as step05_fusao
from scripts.step06_pdf import processar_caso as step06_pdf


def main():

    print("\n=== Ingestão de Casos ===")
    casos = step01_ingestao()

    print("\n=== Inicializando RAG ===")
    vector_db = inicializar_rag()

    for caso_info in casos:

        caso_id = caso_info["caso_id"]
        print("\n======================")
        print(f"PIPELINE: {caso_id}")

        print("\n=== Processamento de Áudio ===")
        step02_audio(caso_info)

        print("\n=== Processamento de Vídeo ===")
        step03_video(caso_info)

        print("\n=== RAG Protocolar ===")
        step04_rag(caso_info, vector_db)

        print("\n=== Fusão Multimodal ===")
        step05_fusao(caso_info)

        print("\n=== Gerando PDF ===")
        step06_pdf(caso_info)

    print("\n=== Pipeline Concluído ===")
 
if __name__ == "__main__":
    main()