from chatbot import Chatbot
from newsclassifier import NewsClassifier
# print("Hi! Tell me a news title and I will define the topic.")
# user_input = input(':')

def extra_title(user_input):
    assisant = Chatbot("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

    system_prompt = 'You will get a messy input, you job is extracted title from the user\'s input.Output ONLY the exact title text from the user\'s input.'

    assisant.add_system_prompt(system_prompt)
    output = assisant.generate_reply(user_input)

    first_sentence = output.split(".")[0].strip()
    return first_sentence

def polish_output(user_input):
    assisant = Chatbot("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    system_prompt = 'You are a friendly chatbot. You will receive a news topic, briefly explain what the main topic of the news is.'
    assisant.set_arguments({'do_sample':False,'max_new_tokens':128})
    assisant.add_system_prompt(system_prompt)
    output = assisant.generate_reply(user_input)
    return output



news_classifier = NewsClassifier("trained_lg_model.pkl")
print('classifier running, enter `exit` to quit')
while True:
    input_text = input('Enter a news headline to classify:')
    if input_text == 'exit':
        break
    news_headline = extra_title(input_text)
    classification = news_classifier.classify_review(news_headline)
    output = polish_output('news_title: ' + news_headline + 'news_topic' + classification)
    print(output)





