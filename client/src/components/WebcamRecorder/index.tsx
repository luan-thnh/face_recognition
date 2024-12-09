'use client';
import React from 'react';
import Webcam from 'react-webcam';

interface Props {
  webcamRef: React.RefObject<Webcam>;
  className?: string;
}

const WebcamRecorder = ({ webcamRef, className }: Props) => {
  return (
    <Webcam
      audio={false}
      ref={webcamRef}
      className={className}
      screenshotFormat="image/jpeg"
      videoConstraints={{
        width: 640,
        height: 360,
        facingMode: 'user',
      }}
    />
  );
};

export default WebcamRecorder;
