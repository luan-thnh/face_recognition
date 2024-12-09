import React, { Suspense, useEffect, useRef, useState } from 'react';
import axios from 'axios';
import Map from '../Map';
import { Button } from '@nextui-org/react';
import toast from 'react-hot-toast';
import { CheckInIcon } from './CheckInIcon';

interface CheckInResponse {
  status: string;
  person_id?: string;
  confidence?: number;
  detail?: string;
}

interface Coordinates {
  latitude: number;
  longitude: number;
}

const CheckIn: React.FC = () => {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const [location, setLocation] = useState<Coordinates | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  const allowedLocation: Coordinates = {
    latitude: 11.940419,
    longitude: 108.458313,
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
    setError(null);

    try {
      const currentLocation = await getCurrentLocation();
      const distance = calculateDistance(currentLocation, allowedLocation);

      if (distance > 1) {
        toast.error('Your current location is too far.');
        setError('Your current location is too far.');
        setLoading(false);
        return;
      }

      // Create a canvas with the same dimensions as the video
      const canvas = document.createElement('canvas');
      const video = videoRef.current;
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;

      const context = canvas.getContext('2d');
      if (!context) {
        throw new Error('Failed to get canvas context');
      }

      // Draw the current frame from the video
      context.drawImage(video, 0, 0, canvas.width, canvas.height);

      // Convert to blob with high quality
      const imageBlob = await new Promise<Blob | null>((resolve) => canvas.toBlob(resolve, 'image/jpeg', 1.0));

      if (!imageBlob) {
        throw new Error('Failed to create image blob');
      }

      const formData = new FormData();
      formData.append('file', imageBlob, 'frame.jpg');

      const response = await axios.post<CheckInResponse>('http://localhost:8000/checkin', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      if (response.data.status === 'error') {
        toast.error(response.data.detail || 'Check-in failed');
        setError(response.data.detail || 'Check-in failed');
      } else {
        toast.success(`Successfully! Welcome, ${response.data.person_id}`);
      }
    } catch (error) {
      console.error('Error during check-in: ', error);
      setError('Failed to check-in. Please try again.');
      toast.error('Failed to check-in. Please try again.');
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
    <div className="flex flex-col items-center justify-center min-h-screen p-4">
      <h1 className="text-3xl font-bold mb-6 text-blue-600 uppercase">Welcome to</h1>
      <div className="relative flex gap-4 rounded-lg">
        <video
          ref={videoRef}
          width="480"
          height="480"
          className="border ring-2 ring-blue-600/60 ring-offset-2 rounded-lg shadow-lg transform -scale-x-100"
        />
        <Suspense>
          <Map latitude={location?.latitude ?? 0} longitude={location?.longitude ?? 0} />
        </Suspense>
      </div>
      <div className="mt-6 space-x-4">
        <Button
          onClick={captureFrame}
          variant="solid"
          color="primary"
          isLoading={loading}
          startContent={<CheckInIcon />}
        >
          Check In
        </Button>
      </div>
      {error && <p className="mt-4 text-red-600">{error}</p>}
    </div>
  );
};

export default CheckIn;
