import React from 'react';
import { Line } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  Filler,
} from 'chart.js';

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  Filler
);

// MOCK DATA - In a real app, this would be an API call to /api/score-history/{symbol}
const generateMockHistory = (baseScore) => {
  let history = [];
  let score = baseScore;
  for (let i = 0; i < 30; i++) {
    score += Math.random() * 4 - 2; // small daily fluctuation
    history.push(Math.round(score));
  }
  return history;
}

const CreditScoreChart = ({ symbol, currentScore }) => {
  const scoreHistory = generateMockHistory(currentScore);
  const labels = Array.from({ length: 30 }, (_, i) => `Day ${i - 29}`);

  const data = {
    labels,
    datasets: [
      {
        label: `${symbol} Credit Score Trend`,
        data: scoreHistory,
        borderColor: 'rgba(41, 98, 255, 0.8)', // --color-accent
        backgroundColor: (context) => {
          const ctx = context.chart.ctx;
          const gradient = ctx.createLinearGradient(0, 0, 0, 200);
          gradient.addColorStop(0, "rgba(41, 98, 255, 0.3)");
          gradient.addColorStop(1, "rgba(41, 98, 255, 0)");
          return gradient;
        },
        tension: 0.3,
        fill: true,
        pointRadius: 0,
      },
    ],
  };

  const options = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        display: false,
      },
    },
    scales: {
      x: {
        grid: {
          color: 'rgba(42, 46, 57, 0.5)', // --color-border
        },
        ticks: {
          color: '#787b86', // --color-text-secondary
          maxRotation: 0,
          autoSkip: true,
          maxTicksLimit: 6,
        },
      },
      y: {
        grid: {
          color: 'rgba(42, 46, 57, 0.5)',
        },
        ticks: {
          color: '#787b86',
        },
      },
    },
  };

  return <Line options={options} data={data} />;
};

export default CreditScoreChart;