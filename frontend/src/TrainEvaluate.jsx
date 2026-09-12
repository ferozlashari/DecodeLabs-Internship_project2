import { useState } from 'react';

function heatColor(value, max) {
  const intensity = max === 0 ? 0 : value / max;
  const alpha = 0.12 + intensity * 0.55;
  return `rgba(232, 163, 61, ${alpha.toFixed(2)})`;
}

export default function TrainEvaluate({ onTrain, result, training, error }) {
  const [k, setK] = useState(5);
  const [testSize, setTestSize] = useState(0.2);

  const maxCell = result ? Math.max(...result.confusion_matrix.flat()) : 0;

  return (
    <>
      <div className="page-head">
        <h1>Train &amp; evaluate</h1>
        <p>
          Trains a K-Nearest Neighbors classifier to predict OrderStatus from the order's
          product, payment, coupon, referral source, quantity, and price details.
        </p>
      </div>

      <div className="row">
        <div className="panel" style={{ flex: '0 0 260px' }}>
          <p className="panel-title">Settings</p>

          <div className="field">
            <label>
              K (neighbors) — <span className="value">{k}</span>
            </label>
            <input
              type="range"
              min="1"
              max="25"
              value={k}
              onChange={(e) => setK(Number(e.target.value))}
            />
          </div>

          <div className="field">
            <label>
              Test set size — <span className="value">{Math.round(testSize * 100)}%</span>
            </label>
            <input
              type="range"
              min="0.1"
              max="0.4"
              step="0.05"
              value={testSize}
              onChange={(e) => setTestSize(Number(e.target.value))}
            />
          </div>

          <button className="btn" disabled={training} onClick={() => onTrain(k, testSize)}>
            {training ? 'Training…' : 'Train model'}
          </button>
        </div>

        <div style={{ flex: 1 }}>
          {error && <div className="error-box">{error}</div>}

          {!result && !error && (
            <div className="panel">
              <p className="muted">Set K and test size, then train to see results here.</p>
            </div>
          )}

          {result && (
            <div className="panel">
              <div className="callout">
                With {result.labels.length} roughly equal classes, random guessing scores
                about {(result.random_guess_baseline * 100).toFixed(0)}%. Compare the accuracy
                below against that baseline, not against 100%.
              </div>

              <div className="metric-row">
                <div className="metric">
                  <div className="label">Accuracy</div>
                  <div className="value">{(result.accuracy * 100).toFixed(1)}%</div>
                  <div className="footnote">
                    {result.test_rows} test rows / {result.train_rows} train rows
                  </div>
                </div>
                <div className="metric">
                  <div className="label">F1 (weighted)</div>
                  <div className="value">{result.f1_weighted.toFixed(3)}</div>
                  <div className="footnote">K = {result.k}</div>
                </div>
              </div>

              <p className="panel-title">Confusion matrix (rows = actual, cols = predicted)</p>
              <div className="matrix-wrap">
                <table className="matrix">
                  <thead>
                    <tr>
                      <th />
                      {result.labels.map((l) => (
                        <th key={l}>{l}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {result.confusion_matrix.map((row, i) => (
                      <tr key={i}>
                        <th className="row-label">{result.labels[i]}</th>
                        {row.map((val, j) => (
                          <td
                            key={j}
                            className={`cell${i === j ? ' diag' : ''}`}
                            style={{ background: heatColor(val, maxCell) }}
                          >
                            {val}
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>

              <details style={{ marginTop: 16 }}>
                <summary className="muted" style={{ cursor: 'pointer' }}>
                  Full classification report
                </summary>
                <pre
                  style={{
                    fontSize: 11.5,
                    color: 'var(--text-dim)',
                    overflowX: 'auto',
                    marginTop: 10,
                  }}
                >
                  {result.classification_report}
                </pre>
              </details>
            </div>
          )}
        </div>
      </div>
    </>
  );
}
