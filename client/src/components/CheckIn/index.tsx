import React, { Suspense, useEffect, useRef, useState } from 'react';
import axios from 'axios';
import Map from '../Map';
import { Button } from '@nextui-org/react';
import toast from 'react-hot-toast';
import { CheckInIcon } from './CheckInIcon';
import Webcam from 'react-webcam';
import WebcamRecorder from '../WebcamRecorder';

interface CheckInResponse {
  message: string;
  student_id: string;
  confidence: number;
}

interface Coordinates {
  latitude: number;
  longitude: number;
}

const CheckIn: React.FC = () => {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const webcamRef = useRef<Webcam>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);

  const [location, setLocation] = useState<Coordinates | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  // const [recordedChunks, setRecordedChunks] = useState<Blob[]>([]);
  const [response, setResponse] = useState<CheckInResponse | null>(null);

  const allowedLocation: Coordinates = {
    latitude: 16.054407,
    longitude: 108.202167,
  };

  const calculateDistance = (coords1: Coordinates, coords2: Coordinates): number => {
    const R = 6371;
    const dLat = ((coords2.latitude - coords1.latitude) * Math.PI) / 180;
    const dLon = ((coords2.longitude - coords1.longitude) * Math.PI) / 180;
    const a =
      Math.sin(dLat / 2) * Math.sin(dLat / 2) +
      Math.cos((coords1.latitude * Math.PI) / 180) *
        Math.cos((coords2.latitude * Math.PI) / 180) *
        Math.sin(dLon / 2) *
        Math.sin(dLon / 2);
    const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
    return R * c;
  };

  const getCurrentLocation = (): Promise<Coordinates> => {
    return new Promise((resolve, reject) => {
      navigator.geolocation.getCurrentPosition(
        (position) => {
          resolve({
            latitude: position.coords.latitude,
            longitude: position.coords.longitude,
          });
        },
        (error) => {
          reject(error.message);
        }
      );
    });
  };

  useEffect(() => {
    getCurrentLocation()
      .then((coords) => {
        setLocation(coords);
        setError(null);
      })
      .catch((err) => {
        setLocation(null);
        setError(err);
      });
  }, []);

  useEffect(() => {
    const startVideo = async (): Promise<void> => {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: true });
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          videoRef.current.play();
        }
      } catch (error) {
        console.error('Error accessing webcam: ', error);
      }
    };

    startVideo();
  }, []);

  const handleCaptureAndUpload = async () => {
    try {
      if (!webcamRef.current || !webcamRef.current.stream) {
        toast.error('Camera access is required');
        return;
      }

      // Check location first
      const currentLocation = await getCurrentLocation();
      const distance = calculateDistance(currentLocation, allowedLocation);

      if (distance > 1) {
        toast.error('You are too far from the check-in location');
        return;
      }

      setLoading(true);
      toast.loading('Recording video...');

      // Start recording
      mediaRecorderRef.current = new MediaRecorder(webcamRef.current.stream, {
        mimeType: 'video/webm',
      });

      const chunks: Blob[] = [];

      mediaRecorderRef.current.ondataavailable = (event) => {
        if (event.data.size > 0) {
          chunks.push(event.data);
        }
      };

      return new Promise((resolve, reject) => {
        mediaRecorderRef.current!.onstop = async () => {
          try {
            if (chunks.length === 0) {
              throw new Error('No video data recorded');
            }

            const videoBlob = new Blob(chunks, { type: 'video/webm' });
            const formData = new FormData();
            formData.append('file', videoBlob, 'video.webm');

            toast.loading('Processing video...');

            const response = await axios.post<CheckInResponse>('http://127.0.0.1:8000/predict_video/', formData, {
              headers: {
                'Content-Type': 'multipart/form-data',
              },
            });

            if (response.data) {
              toast.success(`Welcome ${response.data.student_id}!`);
              console.log(`Welcome ${response.data.student_id}!`);
              setResponse(response.data);
            } else {
              toast.error('No face detected in video');
            }

            resolve(response);
          } catch (error: any) {
            toast.error(error.message || 'An error occurred during check-in');
            reject(error);
          } finally {
            setLoading(false);
            toast.dismiss();
          }
        };

        // Start recording for 5 seconds
        mediaRecorderRef.current!.start();
        setTimeout(() => {
          mediaRecorderRef.current?.stop();
        }, 1000);
      });
    } catch (error: any) {
      toast.error(error.message || 'An error occurred');
      setLoading(false);
      console.error('Check-in error:', error);
    }
  };

  return (
    <div className="flex flex-col items-center justify-center min-h-screen p-4">
      <h1 className="text-3xl font-bold mb-6 text-blue-600 uppercase">Welcome to</h1>
      <div className="relative flex gap-4 rounded-lg">
        <WebcamRecorder
          webcamRef={webcamRef}
          className="w-[480px] h-[360px] border ring-2 ring-blue-600/60 ring-offset-2 rounded-lg shadow-lg transform -scale-x-100"
        />
        <Suspense>
          <Map latitude={location?.latitude ?? 0} longitude={location?.longitude ?? 0} />
        </Suspense>
      </div>
      <div className="mt-6 space-x-4">
        <Button
          onClick={handleCaptureAndUpload}
          variant="solid"
          color="primary"
          isLoading={loading}
          startContent={<CheckInIcon />}
        >
          Check In
        </Button>
      </div>
      {error && <p className="mt-4 text-red-600">{error}</p>}
      {response && (
        <div className="mt-6 bg-white shadow-md rounded-lg p-6 w-96">
          <h2 className="text-lg font-bold mb-2 text-gray-800">Response:</h2>
          <pre className="text-sm text-gray-700 bg-gray-100 p-4 rounded-lg">{JSON.stringify(response, null, 2)}</pre>
        </div>
      )}
    </div>
  );
};

export default CheckIn;
