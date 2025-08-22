import React, { useState, useEffect, useCallback } from 'react';
import axios from 'axios';
import './App.css';

import Watchlist from './components/Watchlist';
import CreditScoreChart from './components/CreditScoreChart';
import ExplanationTab from './components/ExplanationTab';
import DetailsTab from './components/DetailsTab';
import NewsFeedTab from './components/NewsFeedTab';

const WATCHLIST_SYMBOLS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'JPM'];

function App() {
    const [activeSymbol, setActiveSymbol] = useState(WATCHLIST_SYMBOLS[0]);
    const [data, setData] = useState(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState('');
    const [activeTab, setActiveTab] = useState('explanation');

    const fetchData = useCallback(async (symbol) => {
        try {
            setLoading(true);
            setError('');
            // The API call is proxied by Nginx to the backend service.
            const response = await axios.get(`/api/score/${symbol}`);
            setData(response.data);
        } catch (err) {
            console.error("Failed to fetch score data:", err);
            setError(err.response?.data?.detail || "An error occurred while fetching data.");
            setData(null);
        } finally {
            setLoading(false);
        }
    }, []);

    useEffect(() => {
        fetchData(activeSymbol);
    }, [activeSymbol, fetchData]);

    const renderTabContent = () => {
        if (!data) return <p>No data available.</p>;
        switch (activeTab) {
            case 'explanation':
                return <ExplanationTab summary={data.summary} contributions={data.featureContributions} />;
            case 'details':
                return <DetailsTab financialData={data.keyFinancials} />;
            case 'news':
                return <NewsFeedTab newsItems={data.recentNews} />;
            default:
                return null;
        }
    };

    const MainContent = () => {
        if (loading) return <p className="status-text">Loading data for {activeSymbol}...</p>;
        if (error) return <p className="status-text error">{error}</p>;
        if (!data) return <p className="status-text">No data available for {activeSymbol}.</p>;
      
        const scoreColor = data.creditScore > 700 ? 'var(--color-positive)' : data.creditScore < 550 ? 'var(--color-negative)' : 'inherit';
        
        return (
            <>
                <div className="main-header">
                    <h1>{data.company} ({data.symbol})</h1>
                    <div className="score" style={{ color: scoreColor }}>
                        {data.creditScore}
                    </div>
                </div>
                <div className="chart-container">
                    <CreditScoreChart symbol={activeSymbol} currentScore={data.creditScore} />
                </div>
            </>
        );
    };

    return (
        <div className="App">
            <Watchlist
                symbols={WATCHLIST_SYMBOLS}
                activeSymbol={activeSymbol}
                onSelectSymbol={setActiveSymbol}
            />
            <main className="main-content">
                <MainContent />
            </main>
            <div className="info-panel">
                <div className="info-tabs">
                    <button onClick={() => setActiveTab('explanation')} className={`tab-button ${activeTab === 'explanation' ? 'active' : ''}`}>Explanation</button>
                    <button onClick={() => setActiveTab('details')} className={`tab-button ${activeTab === 'details' ? 'active' : ''}`}>Key Financials</button>
                    <button onClick={() => setActiveTab('news')} className={`tab-button ${activeTab === 'news' ? 'active' : ''}`}>Live News</button>
                </div>
                <div className="info-content">
                    {loading ? <p>Loading...</p> : renderTabContent()}
                </div>
            </div>
        </div>
    );
}

export default App;