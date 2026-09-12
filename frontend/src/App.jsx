import { useEffect, useState } from 'react';
import { api } from './api';
import Overview from './Overview';
import TrainEvaluate from './TrainEvaluate';
import Predict from './Predict';
import './App.css';

const PAGES = [
  { id: 'overview', label: 'Overview' },
  { id: 'train', label: 'Train & evaluate' },
  { id: 'predict', label: 'Predict' },
];

export default function App() {
  const [page, setPage] = useState('overview');

  const [summary, setSummary] = useState(null);
  const [sample, setSample] = useState(null);
  const [options, setOptions] = useState(null);
  const [loadError, setLoadError] = useState(null);
  const [loading, setLoading] = useState(true);

  const [trainResult, setTrainResult] = useState(null);
  const [training, setTraining] = useState(false);
  const [trainError, setTrainError] = useState(null);

  const [predictResult, setPredictResult] = useState(null);
  const [predicting, setPredicting] = useState(false);
  const [predictError, setPredictError] = useState(null);

  useEffect(() => {
    Promise.all([api.getSummary(), api.getSample(10), api.getOptions()])
      .then(([s, sa, o]) => {
        setSummary(s);
        setSample(sa);
        setOptions(o);
      })
      .catch((err) => setLoadError(err.message))
      .finally(() => setLoading(false));
  }, []);

  const handleTrain = (k, testSize) => {
    setTraining(true);
    setTrainError(null);
    api
      .train(k, testSize)
      .then((res) => {
        setTrainResult(res);
        setPredictResult(null);
      })
      .catch((err) => setTrainError(err.message))
      .finally(() => setTraining(false));
  };

  const handlePredict = (payload) => {
    setPredicting(true);
    setPredictError(null);
    api
      .predict(payload)
      .then(setPredictResult)
      .catch((err) => setPredictError(err.message))
      .finally(() => setPredicting(false));
  };

  return (
    <div className="shell">
      <nav className="rail">
        <div className="rail-brand">
          <div className="mark">Order Status Classifier</div>
          <div className="sub">DecodeLabs · Project 2</div>
        </div>

        <div className="rail-nav">
          {PAGES.map((p, i) => (
            <button
              key={p.id}
              className={page === p.id ? 'active' : ''}
              onClick={() => setPage(p.id)}
            >
              <span className="num">{String(i + 1).padStart(2, '0')}</span>
              {p.label}
            </button>
          ))}
        </div>

        <div className="rail-footer">
          Supervised learning · K-Nearest Neighbors
          <br />
          Dataset_for_Data_Analytics.xlsx
        </div>
      </nav>

      <main className="main">
        {page === 'overview' && (
          <Overview summary={summary} sample={sample} loading={loading} error={loadError} />
        )}
        {page === 'train' && (
          <TrainEvaluate
            onTrain={handleTrain}
            result={trainResult}
            training={training}
            error={trainError}
          />
        )}
        {page === 'predict' && (
          <Predict
            options={options}
            hasModel={!!trainResult}
            onPredict={handlePredict}
            result={predictResult}
            predicting={predicting}
            error={predictError}
          />
        )}
      </main>
    </div>
  );
}
