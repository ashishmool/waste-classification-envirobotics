import speech_recognition as sr
import subprocess

def listen_for_command():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening for command...")
        audio = recognizer.listen(source)
        try:
            command = recognizer.recognize_google(audio)
            print(f"Command received: {command}")
            if "start" in command.lower():
                # Adjust the command to match your script's path
                subprocess.run(["python", "D:/Artificial Intelligence/waste_classifier.py", "start"])
        except sr.UnknownValueError:
            print("Sorry, I did not understand that.")
        except sr.RequestError as e:
            print(f"Sorry, there was an error with the speech recognition service: {e}")

if __name__ == "__main__":
    listen_for_command()
