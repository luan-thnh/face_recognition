'use client';

import { NextUIProvider } from '@nextui-org/react';
import React from 'react';
import { Toaster } from 'react-hot-toast';

const Providers = ({ children }: { children: React.ReactNode }) => {
  return (
    <NextUIProvider>
      {children}
      <Toaster />
    </NextUIProvider>
  );
};

export default Providers;
