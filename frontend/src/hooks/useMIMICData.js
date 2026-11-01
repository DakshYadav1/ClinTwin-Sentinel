import { useCallback, useEffect, useRef, useState } from 'react';
import {
  fetchMIMICRoster,
  fetchMIMICPatient,
  fetchMIMICTrajectory,
} from '../services/mimicApi';

export function useMIMICData() {
  const [roster, setRoster] = useState(null);
  const [selectedStay, setSelectedStay] = useState(null);
  const [patient, setPatient] = useState(null);
  const [trajectory, setTrajectory] = useState(null);
  const [loading, setLoading] = useState(true);
  const [patientLoading, setPatientLoading] = useState(false);
  const [error, setError] = useState(null);

  const requestId = useRef(0);

  const loadRoster = useCallback(async () => {
    try {
      setLoading(true);
      setError(null);

      const result = await fetchMIMICRoster();
      setRoster(result);

      const firstStay = result?.patients?.[0]?.stay_id;

      if (firstStay != null) {
        setSelectedStay((current) => current ?? firstStay);
      }
    } catch (err) {
      setError(err.message || 'Unable to load MIMIC roster');
    } finally {
      setLoading(false);
    }
  }, []);

  const selectStay = useCallback((stayId) => {
    if (String(stayId) === String(selectedStay)) return;

    setPatient(null);
    setTrajectory(null);
    setSelectedStay(stayId);
    setPatientLoading(true);
    setError(null);
  }, [selectedStay]);

  useEffect(() => {
    if (selectedStay == null) return;

    const currentRequest = ++requestId.current;
    let active = true;

    const loadPatient = async () => {
      setPatientLoading(true);
      setError(null);

      try {
        const [patientResult, trajectoryResult] = await Promise.all([
          fetchMIMICPatient(selectedStay),
          fetchMIMICTrajectory(selectedStay),
        ]);

        if (!active || currentRequest !== requestId.current) return;

        if (
          String(patientResult?.patient?.stay_id) !== String(selectedStay) ||
          String(trajectoryResult?.stay_id) !== String(selectedStay)
        ) {
          throw new Error('Patient response does not match the selected ICU stay.');
        }

        setPatient(patientResult);
        setTrajectory(trajectoryResult);
      } catch (err) {
        if (!active || currentRequest !== requestId.current) return;

        setPatient(null);
        setTrajectory(null);
        setError(err.message || 'Unable to load MIMIC patient data');
      } finally {
        if (active && currentRequest === requestId.current) {
          setPatientLoading(false);
        }
      }
    };

    loadPatient();

    return () => {
      active = false;
    };
  }, [selectedStay]);

  useEffect(() => {
    loadRoster();
  }, [loadRoster]);

  return {
    roster,
    selectedStay,
    selectStay,
    patient,
    trajectory,
    loading,
    patientLoading,
    error,
    refresh: loadRoster,
  };
}
