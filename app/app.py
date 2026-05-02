import gradio as gr
 # Hardcoded stub — replace with real inference in next milestone
def query_papers(question: str) -> str: 
    if not question.strip(): 
        return "Please enter a question." 
    return ( f"[STUB] You asked: '{question}'\n\n" "This response will be replaced by the RAG pipeline " "once fine-tuning is complete." ) 
with gr.Blocks(title="arXiv Paper Q&A") as demo: 
    gr.Markdown("## arXiv LLM — Paper Q&A") 
    gr.Markdown("Ask questions about CS.LG / CS.AI papers (2023–2024).") 
    with gr.Row(): 
        q_box = gr.Textbox( 
            label="Your question", 
            placeholder="What are the key ideas in recent RLHF papers?", 
            lines=2, 
        ) 
        submit_btn = gr.Button("Ask", variant="primary") 
        answer_box = gr.Textbox(label="Answer", lines=6, interactive=False) 
        submit_btn.click(fn=query_papers, inputs=q_box, outputs=answer_box) 
        q_box.submit(fn=query_papers, inputs=q_box, outputs=answer_box) 
if __name__ == "__main__": 
    demo.launch(share=True) # share=True gives a public Colab URL