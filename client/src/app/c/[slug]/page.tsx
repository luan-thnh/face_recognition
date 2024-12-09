'use client';

import CheckIn from '@/components/CheckIn';
import { notFound } from 'next/navigation';
import { useEffect, useState } from 'react';

export default function Home() {
  const [isExpired, setIsExpired] = useState<boolean>(false);

  useEffect(() => {
    const expireTime = document.cookie
      .split('; ')
      .find((row) => row.startsWith('expire-time='))
      ?.split('=')[1];

    if (!expireTime) {
      setIsExpired(true);
      return;
    }

    const expirationTime = parseInt(expireTime, 10);

    if (Date.now() > expirationTime) {
      setIsExpired(true);
    }
  }, []);

  useEffect(() => {
    if (isExpired) {
      notFound();
    }
  }, [isExpired]);

  return (
    <div>
      <CheckIn />
    </div>
  );
}
