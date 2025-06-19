document.addEventListener("DOMContentLoaded", () => {
    const video = document.getElementById('cameraFeed');
    const toggleMic = document.getElementById('toggleMic');
    const toggleVideo = document.getElementById('toggleVideo');
    let stream = null;
    let micEnabled = false;  // Add this line
    
    let videoEnabled = true;

    let recorder = null;
    let isRecording = false;
    let audioChunks = [];
    let stopTimeout;


    // Check if browser supports getUserMedia
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        console.error('getUserMedia is not supported in this browser');
        alert('Your browser does not support camera access. Please try a modern browser like Chrome, Firefox, or Edge.');
    }

    async function startCamera() {
        try {
            // Avoid requesting if both are disabled
            if (!micEnabled && !videoEnabled) {
                console.warn('Both mic and camera are disabled. Skipping media access.');
                return;
            }

            if (window.location.protocol !== 'https:' && window.location.hostname !== 'localhost' && window.location.hostname !== '127.0.0.1') {
                throw new Error('Camera access requires HTTPS or localhost');
            }

            stream = await navigator.mediaDevices.getUserMedia({
                video: videoEnabled ? {
                    width: { ideal: 1280 },
                    height: { ideal: 720 },
                    facingMode: "user"
                } : false,
                audio: micEnabled
            });

            video.srcObject = stream;

            video.onloadedmetadata = () => {
                video.play().then(() => {
                    toggleMic.disabled = false;
                    toggleVideo.disabled = false;

                    toggleMic.classList.toggle('active', micEnabled);
                    toggleMic.classList.toggle('inactive', !micEnabled);

                    toggleVideo.classList.toggle('active', videoEnabled);
                    toggleVideo.classList.toggle('inactive', !videoEnabled);
                });
            };

        } catch (err) {
            console.error('Error accessing camera/microphone:', err);
            alert('Camera/mic access issue: ' + err.message);
        }
    }
       
    function stopCamera() {
        if (stream) {
            stream.getTracks().forEach(track => track.stop());
            video.srcObject = null;
            stream = null;
            toggleMic.disabled = true;
            toggleVideo.disabled = true;
            toggleMic.classList.remove('active', 'inactive');
            toggleVideo.classList.remove('active', 'inactive');
        }
    }
  
    const micLabel = document.getElementById('micLabel');

    toggleMic.addEventListener('click', async () => {
        const micLabel = document.getElementById('micLabel');

        if (isRecording && recorder && recorder.state === "recording") {
            recorder.stop(); // Stop immediately if user taps again
            micLabel.textContent = "🔄 Processing...";
            isRecording = false;
            return;
        }

        micLabel.textContent = "🎧 Listening...";
        micEnabled = true;

        if (!stream || !stream.getAudioTracks().length) {
            await startCamera();
        }

        const audioStream = new MediaStream(stream.getAudioTracks());
        recorder = new MediaRecorder(audioStream);
        audioChunks = [];

        recorder.ondataavailable = e => audioChunks.push(e.data);

        recorder.onstop = async () => {
            const blob = new Blob(audioChunks, { type: 'audio/webm' });
            const reader = new FileReader();
            reader.onloadend = async () => {
                const base64Audio = reader.result.split(',')[1];
                const res = await fetch('/talk', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ audio: base64Audio })
                });

                const js = await res.json();
                document.getElementById('txt').innerText = js.transcript;
                const player = document.getElementById('player');
                player.src = 'data:audio/mp3;base64,' + js.audio;
                player.play();

                micLabel.textContent = "🎙️ Tap to Speak";
                isRecording = false;
            };
            reader.readAsDataURL(blob);
        };

        recorder.start();
        isRecording = true;

        // Optional: Auto-stop after 5s if user doesn’t stop manually
        setTimeout(() => {
            if (isRecording && recorder && recorder.state === "recording") {
                recorder.stop();
                micLabel.textContent = "🔄 Processing...";
                isRecording = false;
            }
        }, 8000);
    });

      
    toggleVideo.addEventListener('click', () => {
        if (stream) {
            const videoTrack = stream.getVideoTracks()[0];
            if (videoTrack) {
                videoTrack.stop(); // stop existing track
            }
        }

        videoEnabled = !videoEnabled;

        if (videoEnabled) {
            startCamera(); // restart with updated flags
        } else {
            if (stream) {
                const remainingTracks = stream.getTracks().filter(track => track.kind !== 'video');
                if (remainingTracks.length === 0) {
                    stopCamera();
                }
            }
            video.srcObject = null;
        }

        toggleVideo.classList.toggle('active', videoEnabled);
        toggleVideo.classList.toggle('inactive', !videoEnabled);
    });
      

    // Start camera automatically
    startCamera();

    // Clean up
    window.addEventListener('beforeunload', () => {
        if (stream) {
            stream.getTracks().forEach(track => track.stop());
        }
    });

    document.addEventListener('visibilitychange', () => {
        if (document.hidden && stream) {
            stopCamera();
        } else if (!document.hidden && !stream && (micEnabled || videoEnabled)) {
            startCamera();  // only restart if at least one is enabled
        }
    });
      
});
  

