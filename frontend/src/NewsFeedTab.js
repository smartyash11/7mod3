import React from 'react';

const NewsFeedTab = ({ newsItems }) => {
    if (!newsItems || newsItems.length === 0) {
        return <p>No recent news found.</p>;
    }

    return (
        <div>
            {newsItems.map((item, index) => (
                <div className="news-item" key={index}>
                    <p className="news-headline">{item.title}</p>
                    <span className="news-source">{item.source}</span>
                </div>
            ))}
        </div>
    );
};

export default NewsFeedTab;