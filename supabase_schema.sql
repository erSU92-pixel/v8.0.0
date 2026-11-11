-- v8.0.0 migration — safe to run on existing v7 table
ALTER TABLE signals ADD COLUMN IF NOT EXISTS signal_id TEXT;
ALTER TABLE signals ADD COLUMN IF NOT EXISTS entry_price NUMERIC;
ALTER TABLE signals ADD COLUMN IF NOT EXISTS status TEXT DEFAULT 'OPEN';
ALTER TABLE signals ADD COLUMN IF NOT EXISTS closed_at TIMESTAMPTZ;
ALTER TABLE signals ADD COLUMN IF NOT EXISTS closed_price NUMERIC;
ALTER TABLE signals ADD COLUMN IF NOT EXISTS max_hold_hours NUMERIC DEFAULT 48;

CREATE UNIQUE INDEX IF NOT EXISTS idx_signals_signal_id
    ON signals(signal_id) WHERE signal_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_signals_status ON signals(status);

-- Add CHECK constraint if not present
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint WHERE conname = 'signals_status_check'
    ) THEN
        ALTER TABLE signals ADD CONSTRAINT signals_status_check
            CHECK (status IN ('OPEN','WIN_TP1','WIN_TP2','LOSS','TIMEOUT','CANCELLED'));
    END IF;
END $$;
