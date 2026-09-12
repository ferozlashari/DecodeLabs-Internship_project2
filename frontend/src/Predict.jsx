import { useState } from 'react';

const STATUS_COLORS = {
  Cancelled: '#e15252',
  Delivered: '#4fb286',
  Pending: '#d9c25a',
  Returned: '#7a8ff0',
  Shipped: '#4fb0d9',
};

const DEFAULTS = {
  Quantity: 2,
  UnitPrice: 25,
  ItemsInCart: 3,
  TotalPrice: 50,
};

export default function Predict({ options, hasModel, onPredict, result, predicting, error }) {
  const [form, setForm] = useState({
    ...DEFAULTS,
    Product: '',
    PaymentMethod: '',
    CouponCode: '',
    ReferralSource: '',
  });

  const ready =
    options &&
    form.Product &&
    form.PaymentMethod &&
    form.CouponCode &&
    form.ReferralSource;

  const update = (key, value) => setForm((f) => ({ ...f, [key]: value }));

  const sortedProba = result?.probabilities
    ? Object.entries(result.probabilities).sort((a, b) => b[1] - a[1])
    : [];

  return (
    <>
      <div className="page-head">
        <h1>Try a prediction</h1>
        <p>Fill in a hypothetical order's details and get a live predicted OrderStatus.</p>
      </div>

      {!hasModel && (
        <div className="callout">Train a model on the "Train &amp; evaluate" page first.</div>
      )}

      <div className="row">
        <div className="panel">
          <p className="panel-title">Order details</p>

          <div className="row">
            <div className="field">
              <label>Quantity</label>
              <input
                type="number"
                min="1"
                value={form.Quantity}
                onChange={(e) => update('Quantity', Number(e.target.value))}
              />
            </div>
            <div className="field">
              <label>Unit price</label>
              <input
                type="number"
                min="0"
                step="0.01"
                value={form.UnitPrice}
                onChange={(e) => update('UnitPrice', Number(e.target.value))}
              />
            </div>
          </div>

          <div className="row">
            <div className="field">
              <label>Items in cart</label>
              <input
                type="number"
                min="1"
                value={form.ItemsInCart}
                onChange={(e) => update('ItemsInCart', Number(e.target.value))}
              />
            </div>
            <div className="field">
              <label>Total price</label>
              <input
                type="number"
                min="0"
                step="0.01"
                value={form.TotalPrice}
                onChange={(e) => update('TotalPrice', Number(e.target.value))}
              />
            </div>
          </div>

          {options &&
            ['Product', 'PaymentMethod', 'CouponCode', 'ReferralSource'].map((field) => (
              <div className="field" key={field}>
                <label>{field}</label>
                <select value={form[field]} onChange={(e) => update(field, e.target.value)}>
                  <option value="" disabled>
                    Select…
                  </option>
                  {options[field].map((opt) => (
                    <option key={opt} value={opt}>
                      {opt}
                    </option>
                  ))}
                </select>
              </div>
            ))}

          <button
            className="btn"
            disabled={!hasModel || !ready || predicting}
            onClick={() => onPredict(form)}
          >
            {predicting ? 'Predicting…' : 'Predict OrderStatus'}
          </button>
        </div>

        <div className="panel">
          <p className="panel-title">Result</p>

          {error && <div className="error-box">{error}</div>}

          {!result && !error && <p className="muted">No prediction yet.</p>}

          {result && (
            <>
              <div className="predict-result">
                <span
                  className="tag"
                  style={{ color: STATUS_COLORS[result.prediction] || 'var(--accent)' }}
                >
                  {result.prediction}
                </span>
              </div>

              {sortedProba.length > 0 && (
                <>
                  <p className="panel-title">Class probabilities (nearest neighbors)</p>
                  {sortedProba.map(([status, p]) => (
                    <div className="proba-row" key={status}>
                      <span className="proba-label">{status}</span>
                      <span className="proba-track">
                        <span
                          className="proba-fill"
                          style={{
                            width: `${p * 100}%`,
                            background: STATUS_COLORS[status] || 'var(--accent)',
                          }}
                        />
                      </span>
                      <span className="proba-value">{(p * 100).toFixed(0)}%</span>
                    </div>
                  ))}
                </>
              )}
            </>
          )}
        </div>
      </div>
    </>
  );
}
