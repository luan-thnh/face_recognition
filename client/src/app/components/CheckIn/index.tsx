import React, { useEffect, useRef, useState } from 'react';
import axios from 'axios';
import dynamic from 'next/dynamic';

interface CheckInResponse {
  status: string;
  person_id?: string;
  confidence?: number;
}

interface Coordinates {
  latitude: number;
  longitude: number;
}

const Map = dynamic(() => import('../Map').then((mod) => mod), { ssr: false });

const CheckIn: React.FC = () => {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const [location, setLocation] = useState<Coordinates | null>(null);
  const [result, setResult] = useState<CheckInResponse | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  const allowedLocation: Coordinates = {
    latitude: 16.0464896,
    longitude: 108.2392576,
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

  const captureFrame = async (): Promise<void> => {
    if (!videoRef.current) return;

    setLoading(true);
    setResult(null);
    setError(null);

    try {
      const currentLocation = await getCurrentLocation();
      const distance = calculateDistance(currentLocation, allowedLocation);

      if (distance > 1) {
        setError('Your current location is too far from the allowed check-in area.');
        setLoading(false);
        return;
      }

      const canvas = document.createElement('canvas');
      const context = canvas.getContext('2d');
      if (!context) {
        console.error('Failed to get canvas context');
        setLoading(false);
        return;
      }

      canvas.width = videoRef.current.videoWidth;
      canvas.height = videoRef.current.videoHeight;
      context.drawImage(videoRef.current, 0, 0, canvas.width, canvas.height);

      const imageBlob = await new Promise<Blob | null>((resolve) => canvas.toBlob(resolve, 'image/jpeg'));
      if (!imageBlob) {
        console.error('Failed to create image blob');
        setLoading(false);
        return;
      }

      const formData = new FormData();
      formData.append('file', imageBlob, 'frame.jpg');

      const response = await axios.post<CheckInResponse>('http://localhost:8000/checkin/', formData);
      setResult(response.data);
    } catch (error) {
      console.error('Error during check-in: ', error);
      setError('Failed to check-in. Please try again.');
    } finally {
      setLoading(false);
    }
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

  return (
    <div className="flex flex-col items-center justify-center min-h-screen bg-gray-100 p-4">
      <h1 className="text-3xl font-bold mb-6 text-gray-800">Real-time Check-in</h1>
      <div className="relative flex">
        <video ref={videoRef} width="480" height="480" className="border rounded-lg shadow-lg transform -scale-x-100" />
        <Map latitude={location?.latitude ?? 0} longitude={location?.longitude ?? 0} />
        {loading && (
          <div className="absolute inset-0 flex items-center justify-center bg-black bg-opacity-50 text-white font-bold text-xl">
            Loading...
          </div>
        )}
      </div>
      <div className="mt-6 space-x-4">
        <button
          onClick={captureFrame}
          className="px-4 py-2 bg-green-500 text-white rounded-lg hover:bg-green-600 focus:outline-none shadow"
        >
          Capture Frame
        </button>
      </div>
      {error && <p className="mt-4 text-red-600">{error}</p>}
      {result && (
        <div className="mt-6 text-center">
          <h2 className="text-2xl font-semibold text-gray-700">Result:</h2>
          {result.status === 'success' ? (
            <p className="text-green-600">
              Welcome, <span className="font-bold">{result.person_id}</span> (Confidence: {result.confidence})
            </p>
          ) : (
            <p className="text-red-600">Failed to recognize face.</p>
          )}
        </div>
      )}
    </div>
  );
};

export default CheckIn;
