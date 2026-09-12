const STATUS_COLORS = {
  Cancelled: 'var(--status-cancelled)',
  Delivered: 'var(--status-delivered)',
  Pending: 'var(--status-pending)',
  Returned: 'var(--status-returned)',
  Shipped: 'var(--status-shipped)',
};

export default function Overview({ summary, sample, loading, error }) {
  if (error) return <div className="error-box">Couldn't load the dataset: {error}</div>;
  if (loading || !summary) return <p className="skeleton">Loading dataset…</p>;

  const maxCount = Math.max(...Object.values(summary.order_status_counts));

  return (
    <>
      <div className="page-head">
        <h1>Dataset overview</h1>
        <p>
          {summary.rows} orders across {summary.columns.length} columns, loaded from{' '}
          <code>Dataset_for_Data_Analytics.xlsx</code>. {summary.missing_values} missing values
          remain after cleaning.
        </p>
      </div>

      <div className="row">
        <div className="panel">
          <p className="panel-title">OrderStatus distribution</p>
          {Object.entries(summary.order_status_counts).map(([status, count]) => (
            <div className="dist-row" key={status}>
              <span className="dist-label">{status}</span>
              <span className="dist-track">
                <span
                  className="dist-fill"
                  style={{
                    width: `${(count / maxCount) * 100}%`,
                    background: STATUS_COLORS[status] || 'var(--accent)',
                  }}
                />
              </span>
              <span className="dist-count">{count}</span>
            </div>
          ))}
        </div>

        <div className="panel">
          <p className="panel-title">Columns</p>
          <p className="muted" style={{ lineHeight: 1.9 }}>
            {summary.columns.join(', ')}
          </p>
        </div>
      </div>

      <div className="panel">
        <p className="panel-title">Sample rows (first {sample?.length || 0})</p>
        <div className="table-scroll">
          <table className="data-table">
            <thead>
              <tr>
                {sample && sample[0] && Object.keys(sample[0]).map((col) => <th key={col}>{col}</th>)}
              </tr>
            </thead>
            <tbody>
              {sample?.map((row, i) => (
                <tr key={i}>
                  {Object.values(row).map((val, j) => (
                    <td key={j}>{String(val)}</td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </>
  );
}
