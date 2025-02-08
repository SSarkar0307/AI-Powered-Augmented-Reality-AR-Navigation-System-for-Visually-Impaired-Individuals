AI-Powered Augmented Reality (AR) Navigation System for Visually Impaired Individuals   
Overview   
This project is an AI-powered Augmented Reality (AR) navigation system designed to assist visually impaired individuals in navigating their surroundings. The system utilizes computer vision, object detection, and real-time audio feedback to detect obstacles, recognize objects, and provide clear path guidance.

Features  
✅ Object Detection & Recognition – Uses YOLOv8 for real-time object detection.  
✅ Distance Estimation – Calculates distance to obstacles to provide relevant alerts.    
✅ Audio Feedback System – Uses text-to-speech (TTS) to guide users via voice commands.  
✅ Real-Time Navigation – Ensures safe movement by directing users to an obstacle-free path.  
✅ Gesture & Voice Commands – Allows users to interact using simple gestures or voice.  
✅ GPS Integration – (Planned for future updates) Provides location-based navigation.  

Technologies Used  
YOLOv8 – Object detection & recognition  
OpenCV – Image processing  
PyTorch – Deep learning framework  
Pyttsx3 – Offline text-to-speech conversion    
SpeechRecognition – Voice command integration  
OpenCV + AI – Distance estimation & obstacle tracking  

INSTALLATION & SETUP  
1️⃣ Clone the Repository  
```
git clone https://github.com/SSarkar0307/AI-Powered-Augmented-Reality-AR-Navigation-System-for-Visually-Impaired-Individuals.git
cd AI-Powered-Augmented-Reality-AR-Navigation-System-for-Visually-Impaired-Individuals

```
2️⃣ Install Dependencies  
```
pip install -r requirements.txt

```

Challenges & Improvements  
🔴 Latency Issues – Optimized model to improve real-time performance.  
🔴 Distance Estimation Accuracy – Refined calculations to reduce false alerts.  
🔴 Voice Delay – Switched to an offline TTS engine for faster responses.  
🔴 Hardware Limitations – Balanced accuracy and speed for smoother execution.  
🔴 Object Recognition – Improved detection for better scene understanding.  

Future Scope  
🔹 Wearable Implementation - Convert this into smart glasses for hands-free navigation  
🔹 GPS-Based Navigation – Implement location-based path guidance.    
🔹 Haptic Feedback – Integrate vibration alerts for obstacle proximity.  
🔹 Cloud-Based Model Updates – Improve detection accuracy over time.  
🔹 Multilingual Support – Extend voice guidance to multiple languages.  

Contributors  
👨‍💻 Sohan Sarkar – Lead Developer & Researcher  
    
Feel free to contribute or reach out for collaboration! 🚀
