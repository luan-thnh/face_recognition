'use client';
import React, { useRef, useState } from 'react';
import Webcam from 'react-webcam';
import axios from 'axios';

const WebcamRecorder = () => {
  const webcamRef = useRef<Webcam>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const [recordedChunks, setRecordedChunks] = useState<Blob[]>([]);
  const [loading, setLoading] = useState(false);
  const [response, setResponse] = useState<any>(null);

  const handleCaptureAndUpload = async () => {
    if (!webcamRef.current || !webcamRef.current.stream) {
      console.error('Webcam stream is not available.');
      return;
    }

    mediaRecorderRef.current = new MediaRecorder(webcamRef.current.stream, {
      mimeType: 'video/webm',
    });

    mediaRecorderRef.current.ondataavailable = (event) => {
      if (event.data.size > 0) {
        setRecordedChunks((prev) => prev.concat(event.data));
      }
    };

    mediaRecorderRef.current.start();

    setTimeout(() => {
      mediaRecorderRef.current?.stop();
    }, 5000);

    mediaRecorderRef.current.onstop = async () => {
      if (recordedChunks.length === 0) {
        alert('No video recorded!');
        return;
      }

      const blob = new Blob(recordedChunks, { type: 'video/webm' });
      const formData = new FormData();
      formData.append('file', blob, 'video.webm');

      try {
        setLoading(true);
        const res = await axios.post('http://127.0.0.1:8000/predict_video/', formData, {
          headers: {
            'Content-Type': 'multipart/form-data',
          },
        });
        setResponse(res.data);
      } catch (error) {
        console.error('Error uploading video:', error);
        setResponse({ message: 'Error uploading video', error });
      } finally {
        setLoading(false);
      }
    };
  };

  return (
    <div className="flex flex-col items-center justify-center min-h-screen bg-black-100">
      <h1 className="text-2xl font-bold mb-4 text-blue-600">Face Recognition with Webcam</h1>
      <div className="bg-white shadow-md rounded-lg p-6 w-96">
        <Webcam
          audio={true}
          ref={webcamRef}
          className="w-full rounded-lg border border-gray-300"
          videoConstraints={{
            width: 1280,
            height: 720,
            facingMode: 'user',
          }}
        />
        <div className="flex justify-center mt-4">
          <button
            onClick={handleCaptureAndUpload}
            className={`px-4 py-2 text-white font-semibold rounded-lg ${
              loading ? 'bg-gray-500 cursor-not-allowed' : 'bg-blue-500 hover:bg-blue-700'
            }`}
            disabled={loading}
          >
            {loading ? 'Uploading...' : 'Submit'}
          </button>
        </div>
      </div>

      {response && (
        <div className="mt-6 bg-white shadow-md rounded-lg p-6 w-96">
          <h2 className="text-lg font-bold mb-2 text-gray-800">Response:</h2>
          <pre className="text-sm text-gray-700 bg-gray-100 p-4 rounded-lg">{JSON.stringify(response, null, 2)}</pre>
        </div>
      )}
    </div>
  );
};

export default WebcamRecorder;
