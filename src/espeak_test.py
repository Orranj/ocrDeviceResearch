# from espeakng import ESpeakNG

# esng = ESpeakNG()

# try:
#     esng.voice = "english-us"
#     esng.say("hello")
#     # print(esng.voices)
# except Exception as e:
#     print(e)

import espeakng

esng = espeakng.Speaker()
# esng.pitch = 70
esng.wpm = 140
esng.amplitude = 100
esng.voice = "en-us"
esng.say("Purple burglar alarm")
