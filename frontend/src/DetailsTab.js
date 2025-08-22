import React from 'react';

const DetailItem = ({ label, value }) => (
  <div className="detail-item">
    <span className="detail-label">{label}</span>
    <span className="detail-value">{value || 'N/A'}</span>
  </div>
);

const DetailsTab = ({ financialData }) => {
  if (!financialData) return <p>No financial details available.</p>;

  // Function to format large numbers
  const formatNumber = (numStr) => {
    const num = parseInt(numStr);
    if (isNaN(num)) return 'N/A';
    if (num >= 1e12) return `${(num / 1e12).toFixed(2)}T`;
    if (num >= 1e9) return `${(num / 1e9).toFixed(2)}B`;
    if (num >= 1e6) return `${(num / 1e6).toFixed(2)}M`;
    return num.toString();
  };

  return (
    <div className="details-grid">
      <DetailItem label="Market Cap" value={formatNumber(financialData.MarketCapitalization)} />
      <DetailItem label="P/E Ratio" value={financialData.PERatio} />
      <DetailItem label="Return on Equity (TTM)" value={`${(parseFloat(financialData.ReturnOnEquityTTM) * 100).toFixed(2)}%`} />
      <DetailItem label="Debt to Equity Ratio" value={financialData.DebtToEquityRatio} />
      <DetailItem label="EBITDA" value={formatNumber(financialData.EBITDA)} />
      <DetailItem label="Analyst Target Price" value={financialData.AnalystTargetPrice} />
    </div>
  );
};

export default DetailsTab;