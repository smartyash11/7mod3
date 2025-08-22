import React from 'react';

const Watchlist = ({ symbols, activeSymbol, onSelectSymbol }) => {
    return (
        <aside className="watchlist-container">
            <div className="watchlist-header">
                <h2>Watchlist</h2>
            </div>
            <ul className="watchlist-items">
                {symbols.map(symbol => (
                    <li
                        key={symbol}
                        className={`watchlist-item ${symbol === activeSymbol ? 'active' : ''}`}
                        onClick={() => onSelectSymbol(symbol)}
                    >
                        <span className="item-symbol">{symbol}</span>
                    </li>
                ))}
            </ul>
        </aside>
    );
};

export default Watchlist;