'use client';

import { Spinner } from '@nextui-org/react';
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
      <div className="flex justify-center items-center h-screen">
        <Spinner size="lg" />
      </div>
    );
  }

  return (
    <div className="max-w-6xl mx-auto my-8 p-4 bg-white shadow-lg rounded-lg">
      <h1 className="text-2xl font-bold text-blue-600 mb-6 text-center">Check-in History</h1>
      <div className="overflow-x-auto">
        <table className="table-auto w-full border-collapse border border-blue-200">
          <thead>
            <tr className="bg-blue-600 text-white">
              <th className="px-4 py-2 border border-blue-200">ID</th>
              <th className="px-4 py-2 border border-blue-200">Họ và tên</th>
              {Object.keys(checkinHistory[0]?.checkins || {}).map((key) => (
                <th key={key} className="px-4 py-2 border border-blue-200">
                  {key}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {checkinHistory.map((student, index) => (
              <tr key={index} className={`hover:bg-blue-50 ${index % 2 === 0 ? 'bg-blue-100' : 'bg-white'}`}>
                <td className="px-4 py-2 border border-blue-200 text-center text-blue-900">{student.id}</td>
                <td className="px-4 py-2 border border-blue-200 text-center text-blue-900">{student.name}</td>
                {Object.values(student?.checkins || {}).map((value, idx) => (
                  <td key={idx} className="px-4 py-2 border border-blue-200 text-center text-blue-900">
                    {value || '-'}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
};

export default CheckinTable;
