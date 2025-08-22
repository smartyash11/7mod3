import React from 'react';

const ExplanationTab = ({ summary, contributions }) => {
    if (!summary || !contributions) {
        return <p>No explanation data available.</p>;
    }

    return (
        <div>
            <p className="explanation-summary">{summary}</p>
            <ul className="contributions-list">
                {Object.entries(contributions).map(([feature, value]) => (
                    <li key={feature} className={value >= 0 ? 'positive' : 'negative'}>
                        <span className="feature-name">{feature.replace(/([A-Z])/g, ' $1')}</span>
                        <span className="feature-value">{value.toFixed(4)}</span>
                    </li>
                ))}
            </ul>
        </div>
    );
};

export default ExplanationTab;