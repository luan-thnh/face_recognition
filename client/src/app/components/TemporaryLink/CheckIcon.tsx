import React from 'react';
import type { SVGProps } from 'react';

export function CheckIcon(props: SVGProps<SVGSVGElement>) {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" width="1em" height="1em" viewBox="0 0 24 24" {...props}>
      <mask id="lineMdCheckAll0">
        <g
          fill="none"
          stroke="#fff"
          strokeDasharray={24}
          strokeDashoffset={24}
          strokeLinecap="round"
          strokeLinejoin="round"
          strokeWidth={2}
        >
          <path d="M2 13.5l4 4l10.75 -10.75">
            <animate fill="freeze" attributeName="stroke-dashoffset" dur="0.4s" values="24;0"></animate>
          </path>
          <path stroke="#000" strokeWidth={6} d="M7.5 13.5l4 4l10.75 -10.75">
            <animate fill="freeze" attributeName="stroke-dashoffset" begin="0.4s" dur="0.4s" values="24;0"></animate>
          </path>
          <path d="M7.5 13.5l4 4l10.75 -10.75">
            <animate fill="freeze" attributeName="stroke-dashoffset" begin="0.4s" dur="0.4s" values="24;0"></animate>
          </path>
        </g>
      </mask>
      <rect width={24} height={24} fill="currentColor" mask="url(#lineMdCheckAll0)"></rect>
    </svg>
  );
}
