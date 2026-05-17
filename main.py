from scripts import step07_pdf
from scripts.step01_ingestao import processar_casos as step01_ingestao
from scripts.step02_prontuario import processar_caso as step02_prontuario
from scripts.step03_audio import processar_caso as step03_audio
from scripts.step04_video import processar_caso as step04_video
from scripts.step05_rag import processar_caso as step05_rag, inicializar_rag
from scripts.step06_fusao import processar_caso as step06_fusao
from scripts.step07_pdf import processar_caso as step07_pdf


def main():

    print("\n=== Ingestão de Casos ===")
    casos = step01_ingestao()

    print("\n=== Inicializando RAG ===")
    vector_db = inicializar_rag()

    for caso_info in casos:

        caso_id = caso_info["caso_id"]
        print("\n======================")
        print(f"PIPELINE: {caso_id}")

        print("\n=== Análise de Prontuário ===")
        step02_prontuario(caso_info)

        print("\n=== Processamento de Áudio ===")
        step03_audio(caso_info)

        print("\n=== Processamento de Vídeo ===")
        step04_video(caso_info)

        print("\n=== RAG Protocolar ===")
        step05_rag(caso_info, vector_db)

        print("\n=== Fusão Multimodal ===")
        step06_fusao(caso_info)

        print("\n=== Gerando PDF ===")
        step07_pdf(caso_info)

    print("\n=== Pipeline Concluído ===")
 
if __name__ == "__main__":
    main()