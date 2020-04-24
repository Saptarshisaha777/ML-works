import speech_recognition as sr

import webbrowser as wb


r_b = sr.Recognizer()
r_a = sr.Recognizer()


with sr.Microphone() as source:
    print('[Search Google(say GOOGLE to activate)]')
    print('Speak Now')
    audio = r_a.listen(source)

    text = r_a.recognize_google(audio)
    print(text)

    if 'Google' in text:
        url = "https://www.google.com/search?client=firefox-b-d&q="
        with sr.Microphone() as source:
            print('Enter your Search Request:')
            audio = r_b.listen(source)

            try:
                req = r_b.recognize_google(audio)
                wb.get().open_new(url+req)
            except sr.UnknownValueError:
                print('ERROR!!!')
            except sr.RequestError as r:
                print('failure'.format(r))
