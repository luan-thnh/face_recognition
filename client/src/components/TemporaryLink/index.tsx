import React, { useEffect, useState } from 'react';
import { Button } from '@nextui-org/react';
import { cn, generateRandomLink } from '@/utils';
import { useCopy } from '@/hooks/useCopy';
import { CheckIcon } from './CheckIcon';
import { CopyIcon } from './CopyIcon';

const TemporaryLink: React.FC = () => {
  const [temporaryLink, setTemporaryLink] = useState<string | null>(null);
  const [timeRemaining, setTimeRemaining] = useState<number>(0);
  const [isLinkActive, setIsLinkActive] = useState<boolean>(false);

  const [copied, copy] = useCopy();

  const createTemporaryLink = () => {
    const newLink = generateRandomLink();
    setTemporaryLink(newLink);
    setIsLinkActive(true);

    const expireTime = Date.now() + 5 * 60 * 1000; // Hết hạn sau 5 phút
    document.cookie = `expire-time=${expireTime}; path=/`;

    setTimeRemaining(300); // 5p

    const interval = setInterval(() => {
      setTimeRemaining((prevTime) => {
        if (prevTime <= 1) {
          clearInterval(interval);
          setIsLinkActive(false);
          setTemporaryLink(null);
          return 0;
        }
        return prevTime - 1;
      });
    }, 1000);
  };

  useEffect(() => {
    if (timeRemaining <= 0) {
      setIsLinkActive(false);
      setTemporaryLink(null);
    }
  }, [timeRemaining]);

  return (
    <div className="flex flex-col items-center justify-center min-h-screen p-4">
      <h1 className="text-3xl font-bold mb-6 text-blue-600 uppercase">Temporary Link</h1>

      <div className="mt-6 space-x-4">
        <Button onClick={createTemporaryLink} variant="solid" color="primary" disabled={isLinkActive}>
          Create Temporary Link
        </Button>
      </div>

      {temporaryLink && (
        <div className="flex">
          <div className="mt-4">
            <div className={cn(copied ? 'text-green-600' : 'text-blue-600')}>
              <div
                className={cn(
                  'flex items-center gap-4 transition-all py-2 px-4 rounded-lg',
                  copied ? 'bg-green-200/20' : 'bg-blue-300/20'
                )}
              >
                <a href={temporaryLink} target="_blank" rel="noopener noreferrer">
                  {temporaryLink}
                </a>
                <Button
                  onClick={() => copy(temporaryLink)}
                  disabled={copied}
                  variant="light"
                  color={copied ? 'success' : 'primary'}
                  isIconOnly
                >
                  {copied ? <CheckIcon className="w-6 h-6" /> : <CopyIcon className="w-6 h-6" />}
                </Button>
              </div>
            </div>
            <p className="text-red-600 mt-2">
              Time remaining:{' '}
              {Math.floor(timeRemaining / 60)
                .toString()
                .padStart(2, '0')}
              :{(timeRemaining % 60).toString().padStart(2, '0')}
            </p>
          </div>
        </div>
      )}
    </div>
  );
};

export default TemporaryLink;
