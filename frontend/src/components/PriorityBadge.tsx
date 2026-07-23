import { clsx } from 'clsx';

interface Props {
  priority: 'CRITICAL' | 'HIGH' | 'MEDIUM' | 'LOW';
}

export function PriorityBadge({ priority }: Props) {
  let colors = '';
  switch (priority) {
    case 'CRITICAL':
      colors = 'bg-red-100 text-red-800 border-red-200';
      break;
    case 'HIGH':
      colors = 'bg-orange-100 text-orange-800 border-orange-200';
      break;
    case 'MEDIUM':
      colors = 'bg-yellow-100 text-yellow-800 border-yellow-200';
      break;
    case 'LOW':
      colors = 'bg-green-100 text-green-800 border-green-200';
      break;
  }

  return (
    <span className={clsx("inline-flex items-center px-2 py-0.5 rounded text-xs font-semibold border", colors)}>
      {priority}
    </span>
  );
}
