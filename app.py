# from transformers import WhisperProcessor, WhisperForConditionalGeneration
import gradio as gr 
import time
import whisper
import os
from dotenv import load_dotenv
from langchain.chat_models import AzureChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain


dotenv_path = "key.env"
load_dotenv(dotenv_path)

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
OPENAI_API_TYPE  = os.getenv('OPENAI_API_TYPE')
OPENAI_API_BASE = os.getenv('OPENAI_API_BASE')
OPENAI_API_VERSION = os.getenv('OPENAI_API_VERSION')
DEPLOYMENT_NAME = os.getenv('DEPLOYMENT_NAME')

llm_ques_gen_pipeline = AzureChatOpenAI(
    temperature = 0,
    openai_api_base=OPENAI_API_BASE,
    openai_api_version=OPENAI_API_VERSION,
    deployment_name=DEPLOYMENT_NAME,
    openai_api_key=OPENAI_API_KEY,
    openai_api_type=OPENAI_API_TYPE,
)

questions = { "0": {
    "text": "Hi, I hope you're doing well!",
    "type": "answer should be either yes or no",
},"1": {
    "text": "I am very happy to see you here. I understand that after a stroke, it might feel like you've lost a piece of your life as you knew it, and it's entirely normal to experience a whirlwind of emotions like shock, denial, anger, grief, and guilt. Navigating through these feelings can be tough, but acknowledging them and seeking support can make a significant difference. I would love to hear from you, whether you have experienced any sad feelings or stress in the past few days?",
    "type": "answer should be either yes or no",
},"2": {
    "text": "I am sorry you have experienced such feelings. How often would you say you have experienced such feelings?",
    "type": "answer should be an interger between 1 and 5",
}, "3": {
    "text": "oh I am sorry, would you like to share with me in more detail some of the times that you experienced such stressful feelings so I can better provide guidance?",
    "type": "answer is a text. e.g. I felt stressed due to multitasking during meal preparation for my children. first look through the answer and make sure there is no mistake in text to speech recognition. At the end provide the answer in the format of trigger: thought: feeling: behavior:",
}, "4": {
    "text": "Please confirm if I have correctly captured your situation in a helpful manner that would allow us to use a reflective stress management technique?",
    "type": "answer should be either yes or no",
}, "5": {
    "text": "What is an alternetive line of thinking you would propose that can provoke more positive feelings and behavior?",
    "type": "provide suggestions for the user based on their previous answer. Also, provide some useful links at the end as well.",
}}


prompt_template = """
You are interviewing a post stroke patient and asking a series of questions.
here is the question we ask the user:
------------
{question}
------------
The patient has difficulty speaking and we are using a speech to text model to get the user's response. Look through the answer and make sure there is no mistake in text to speech recognition.
The expected output format is {type} so convert the answer to the proper output format based on users response.
Here is the answer we received from interviewee:
------------
{answer}
------------
REFINED ANSWER:
"""

PROMPT_QUESTIONS = PromptTemplate(template=prompt_template, input_variables=["question", "type", "answer"])

# Analyze the text with GPT-4
def analyze(llm, prompt, question, type, answer = ''):
    chain = LLMChain(llm=llm, prompt=prompt)
    refined_answer  = chain.run({'question': question, 'type': type, 'answer': answer})
    return refined_answer

# speech to text
def inference(audio):
    # load audio and pad/trim it to fit 30 seconds
    model = whisper.load_model("tiny.en")
    result = model.transcribe(audio)
    return result["text"]

# Initialize a counter to keep track of the question number
question_counter = [0]
ans = ''
def survey_bot(user_response):
    
    
    if question_counter[0] == 0:
        # Bot's turn to ask a question
        ans = questions[str(question_counter[0])]['text']
        question_counter[0] += 1
        return ans
    
    elif question_counter[0] == 1:
        # Bot's turn to ask a question
        ans = questions[str(question_counter[0])]['text']
        question_counter[0] += 1
        return ans
    
    elif question_counter[0] == 2:
        # Bot's turn to ask a question
        user_response = inference(user_response)
        ans = analyze(llm_ques_gen_pipeline, PROMPT_QUESTIONS, questions[str(question_counter[0]-1)]['text'], questions[str(question_counter[0]-1)]['type'], user_response)
        question_counter[0] += 1
        return f"Seems like you responded as {ans}, {questions[str(question_counter[0]-1)]['text']}"
    
    elif question_counter[0] == 3:
        # Bot's turn to ask a question
        user_response = inference(user_response)
        ans = analyze(llm_ques_gen_pipeline, PROMPT_QUESTIONS, questions[str(question_counter[0]-1)]['text'], questions[str(question_counter[0]-1)]['type'], user_response)
        question_counter[0] += 1
        return f"So on scale of 1-5, your stress level has been {ans}, {questions[str(question_counter[0]-1)]['text']}"
    
    elif question_counter[0] == 4:
        # Bot's turn to ask a question
        global ans_imp
        ans_imp = inference(user_response)
        user_response = ans_imp
        ans = analyze(llm_ques_gen_pipeline, PROMPT_QUESTIONS, questions[str(question_counter[0]-1)]['text'], questions[str(question_counter[0]-1)]['type'], user_response)
        question_counter[0] += 1
        return f"{questions[str(question_counter[0]-1)]['text']}: \n\n {ans}"
    
    else:
        user_response = inference(user_response)
        ans = analyze(llm_ques_gen_pipeline, PROMPT_QUESTIONS, questions[str(question_counter[0]-2)]['text'], questions[str(question_counter[0]-1)]['type'], user_response)
        ans1 = analyze(llm_ques_gen_pipeline, PROMPT_QUESTIONS, questions[str(question_counter[0])]['text'], questions[str(question_counter[0])]['type'], ans_imp)
        # question_counter[0] += 1
        return f"Seems like you responded as {ans}, {ans1}"
    

    # Increment the counter to prepare for the next question

# Define the Gradio interface
iface = gr.Interface(
    fn=survey_bot,                         # Function to be called
    inputs=[
        gr.Audio(source="microphone", type="filepath")],                       # Type of input
    outputs="text",                        # Type of output                             # Enable live updates
    title="Post Stroke Assessment Tool",   # Description of the interface
    initial_value=f"{questions['1']}",       # Initial value to display
)

# Launch the interface
iface.launch()