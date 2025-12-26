from src.models.chatbot import Chatbot
from src.models.newsclassifier import NewsClassifier
from src.utils.get_path import get_path


def extra_title(user_input):
    assisant = Chatbot("Qwen/Qwen2.5-3B-Instruct")

    system_prompt = 'You will get a messy input, you job is extracted news title from the user\'s input.Output ONLY the exact title text from the user\'s input. Ensure the full title is preserved.'
    assisant.set_arguments({'do_sample':False,'max_new_tokens':64})
    assisant.modify_system_prompt(system_prompt)
    output = assisant.generate_reply(user_input)

    return output

def polish_output(user_input):
    assisant = Chatbot("Qwen/Qwen2.5-3B-Instruct")
    system_prompt = '''
    You are a friendly assistant who explains the result of a news classification in natural language.

    You will receive three fields: raw_input, news_title, and news_topic.

    Your task:
    Provide a concise explanation of the classification using and explicitly including the exact words from news_topic.

    Strict rules:

    The response must include the exact term news_topic as given.

    Base your explanation only on news_topic.

    Do NOT invent, infer, or expand any news content or story.

    Do NOT add background information or examples.

    Keep the explanation short, neutral, and factual.
    '''
    assisant.set_arguments({'do_sample':False,'max_new_tokens':64})
    assisant.modify_system_prompt(system_prompt)
    output = assisant.generate_reply(user_input)
    return output



ROOT = get_path(2)
save_model_path = ROOT/"src"/'saved_model'
news_classifier = NewsClassifier(save_model_path/"trained_lg_model.pkl",save_model_path/'mapping.pkl',save_model_path/'vectorizer.pkl' )
print('classifier running, enter `exit` to quit')
while True:
    input_text = input('Enter a news headline to classify:')
    if input_text == 'exit':
        break
    news_headline = extra_title(input_text)
    print(news_headline)
    classification = news_classifier.classify_review(news_headline)
    print(classification)
    output = polish_output('raw_input: '+ input_text + ' news_topic:' + classification)
    print(output)





