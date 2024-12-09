'use client';

import { Spinner, Table, TableBody, TableCell, TableColumn, TableHeader, TableRow } from '@nextui-org/react';
import { useEffect, useState } from 'react';

type CheckinHistory = {
  id: string;
  name: string;
  checkins: Record<string, string>; // Dynamic keys for checkin dates
};

const CheckinTable = () => {
  const [checkinHistory, setCheckinHistory] = useState<CheckinHistory[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchCheckinHistory = async () => {
      try {
        const res = await fetch('http://127.0.0.1:8000/checkin_history');
        const data = await res.json();
        setCheckinHistory(data);
      } catch (error) {
        console.error('Failed to fetch checkin history:', error);
      } finally {
        setLoading(false);
      }
    };

    fetchCheckinHistory();
  }, []);

  if (loading) {
    return (
      <div className="flex justify-center items-center">
        <Spinner size="lg" />
      </div>
    );
  }

  return (
    <div className="max-w-4xl mx-auto my-6">
      <Table aria-label="Check-in History" css={{ height: 'auto', minWidth: '100%' }}>
        <TableHeader>
          <TableColumn>ID</TableColumn>
          <TableColumn>Họ và tên</TableColumn>
          {Object.keys(checkinHistory[0]?.checkins || {}).map((key) => (
            <TableColumn key={key}>{key}</TableColumn>
          ))}
        </TableHeader>
        <TableBody>
          {checkinHistory?.map((student, index) => (
            <TableRow key={index}>
              <TableCell className="text-black">{student.id}</TableCell>
              <TableCell className="text-black">{student.name}</TableCell>
              {Object.values(student?.checkins || {}).map((value, idx) => (
                <TableCell key={idx} className="text-black">
                  {value || '-'}
                </TableCell>
              ))}
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </div>
  );
};

export default CheckinTable;
