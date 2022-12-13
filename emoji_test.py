from transformers import pipeline
import emoji
import googletrans

classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", return_all_scores=True)

change = {'anger' : ":face_with_symbols_on_mouth:",
          'disgust' : ":nauseated_face:",
          'fear' : ":fearful_face:",
          'joy' : ":grinning_face:",
          'neutral' : ":neutral_face:",
          'sadness' : ":loudly_crying_face:",
          'surprise' : ":astonished_face:"
          }

translator = googletrans.Translator()

sentence = "이 학과에는 슬픈 비밀이 있어요.. "

ko_en = translator.translate(sentence, dest='en', src='ko')

emotion= classifier(ko_en.text)
top_emotion = ""
top_score = 0
for num in range(len(emotion[0])):
  if top_score < emotion[0][num]['score'] :
    top_emotion = emotion[0][num]['label']
    top_score = emotion[0][num]['score']

print(f"{sentence}{emoji.emojize(change[top_emotion])}")
#%%
for i in emotion[0]:
    print(i)