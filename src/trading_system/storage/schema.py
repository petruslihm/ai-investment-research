"""DuckDB schema DDL for foundation and V1 tables."""

from __future__ import annotations

SCHEMA_VERSION = 7

DDL_STATEMENTS: tuple[str, ...] = (
    """
    CREATE TABLE IF NOT EXISTS schema_meta (
        key VARCHAR PRIMARY KEY,
        value VARCHAR NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS writer_lease (
        lease_name VARCHAR PRIMARY KEY,
        owner_token VARCHAR NOT NULL,
        acquired_at TIMESTAMPTZ NOT NULL,
        heartbeat_at TIMESTAMPTZ NOT NULL,
        expires_at TIMESTAMPTZ NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS job_lease (
        job_key VARCHAR PRIMARY KEY,
        owner_token VARCHAR NOT NULL,
        status VARCHAR NOT NULL,
        acquired_at TIMESTAMPTZ NOT NULL,
        heartbeat_at TIMESTAMPTZ NOT NULL,
        expires_at TIMESTAMPTZ NOT NULL,
        payload_json VARCHAR
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS issuers (
        issuer_id VARCHAR PRIMARY KEY,
        display_name VARCHAR,
        cik VARCHAR,
        created_at TIMESTAMPTZ NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS instruments (
        instrument_id VARCHAR PRIMARY KEY,
        issuer_id VARCHAR NOT NULL,
        asset_class VARCHAR NOT NULL,
        display_name VARCHAR,
        created_at TIMESTAMPTZ NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS instrument_aliases (
        instrument_id VARCHAR NOT NULL,
        provider VARCHAR NOT NULL,
        symbol VARCHAR NOT NULL,
        effective_from DATE NOT NULL,
        effective_to DATE,
        notes VARCHAR,
        PRIMARY KEY (provider, symbol, effective_from)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS decision_epochs (
        decision_epoch_id VARCHAR PRIMARY KEY,
        created_at TIMESTAMPTZ NOT NULL,
        preprocess_version VARCHAR NOT NULL,
        ensemble_weights_hash VARCHAR NOT NULL,
        thresholds_hash VARCHAR NOT NULL,
        formula_versions_json VARCHAR NOT NULL,
        parent_epoch_id VARCHAR,
        fingerprint VARCHAR NOT NULL,
        manifest_json VARCHAR NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS online_update_ledger (
        model_stream VARCHAR NOT NULL,
        target_version VARCHAR NOT NULL,
        prediction_label_id VARCHAR NOT NULL,
        parent_epoch_id VARCHAR NOT NULL,
        update_batch_hash VARCHAR NOT NULL,
        status VARCHAR NOT NULL,
        created_at TIMESTAMPTZ NOT NULL,
        metadata_json VARCHAR,
        PRIMARY KEY (model_stream, target_version, prediction_label_id, update_batch_hash)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS feature_snapshots (
        feature_snapshot_id VARCHAR PRIMARY KEY,
        price_basis VARCHAR NOT NULL,
        adjustment_revision VARCHAR NOT NULL,
        evaluation_basis VARCHAR NOT NULL,
        provider VARCHAR,
        published_at TIMESTAMPTZ NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS outcome_snapshots (
        outcome_snapshot_id VARCHAR PRIMARY KEY,
        feature_snapshot_id VARCHAR NOT NULL,
        price_basis VARCHAR NOT NULL,
        adjustment_revision VARCHAR NOT NULL,
        evaluation_basis VARCHAR NOT NULL,
        provider VARCHAR,
        published_at TIMESTAMPTZ NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS request_sets (
        request_set_id VARCHAR PRIMARY KEY,
        created_at TIMESTAMPTZ NOT NULL,
        notes VARCHAR
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS ticks (
        tick_id VARCHAR PRIMARY KEY,
        decision_epoch_id VARCHAR,
        feature_snapshot_id VARCHAR,
        committed_at TIMESTAMPTZ NOT NULL,
        status VARCHAR NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS tick_observations (
        tick_id VARCHAR NOT NULL,
        instrument_id VARCHAR NOT NULL,
        payload_json VARCHAR NOT NULL,
        PRIMARY KEY (tick_id, instrument_id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS tick_predictions (
        tick_id VARCHAR NOT NULL,
        instrument_id VARCHAR NOT NULL,
        payload_json VARCHAR NOT NULL,
        PRIMARY KEY (tick_id, instrument_id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS tick_recommendations (
        tick_id VARCHAR NOT NULL,
        recommendation_id VARCHAR NOT NULL,
        source VARCHAR NOT NULL,
        payload_json VARCHAR NOT NULL,
        PRIMARY KEY (tick_id, recommendation_id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS watermarks (
        watermark_key VARCHAR PRIMARY KEY,
        instrument_id VARCHAR,
        value VARCHAR NOT NULL,
        updated_at TIMESTAMPTZ NOT NULL,
        tick_id VARCHAR NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS alert_outbox (
        outbox_id VARCHAR PRIMARY KEY,
        alert_id VARCHAR NOT NULL,
        channel VARCHAR NOT NULL,
        idempotency_key VARCHAR NOT NULL UNIQUE,
        status VARCHAR NOT NULL,
        attempts INTEGER NOT NULL,
        last_error VARCHAR,
        created_at TIMESTAMPTZ NOT NULL,
        updated_at TIMESTAMPTZ NOT NULL,
        payload_json VARCHAR
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS portfolio_lots (
        lot_id VARCHAR PRIMARY KEY,
        instrument_id VARCHAR NOT NULL,
        acquisition_units DOUBLE NOT NULL,
        acquisition_price DOUBLE NOT NULL,
        acquired_on DATE NOT NULL,
        notes VARCHAR
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS portfolio_meta (
        key VARCHAR PRIMARY KEY,
        value VARCHAR NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS btc_sleeve_state (
        id INTEGER PRIMARY KEY CHECK (id = 1),
        instrument_id VARCHAR NOT NULL,
        symbol VARCHAR NOT NULL,
        liquidity VARCHAR NOT NULL,
        current_units DOUBLE NOT NULL,
        recommended_units DOUBLE,
        last_action VARCHAR NOT NULL,
        transfer_caveat VARCHAR,
        updated_at TIMESTAMPTZ NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS runtime_events (
        event_id VARCHAR PRIMARY KEY,
        kind VARCHAR NOT NULL,
        status VARCHAR NOT NULL,
        message VARCHAR NOT NULL,
        progress_done INTEGER,
        progress_total INTEGER,
        duration_ms DOUBLE,
        created_at TIMESTAMPTZ NOT NULL,
        metadata_json VARCHAR
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS data_health_snapshots (
        snapshot_id VARCHAR PRIMARY KEY,
        overall VARCHAR NOT NULL,
        equity_feed VARCHAR NOT NULL,
        btc_feed VARCHAR NOT NULL,
        coverage_json VARCHAR NOT NULL,
        last_tick_at TIMESTAMPTZ,
        created_at TIMESTAMPTZ NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS model_change_journal (
        event_id VARCHAR PRIMARY KEY,
        kind VARCHAR NOT NULL,
        payload_json VARCHAR NOT NULL,
        created_at TIMESTAMPTZ NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS equity_daily_bars (
        instrument_id VARCHAR NOT NULL,
        session_date DATE NOT NULL,
        open DOUBLE NOT NULL,
        high DOUBLE NOT NULL,
        low DOUBLE NOT NULL,
        close DOUBLE NOT NULL,
        volume DOUBLE NOT NULL,
        finality VARCHAR NOT NULL,
        adjustment_revision VARCHAR NOT NULL,
        provider VARCHAR NOT NULL,
        receive_ts TIMESTAMPTZ NOT NULL,
        PRIMARY KEY (instrument_id, session_date, adjustment_revision)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS btc_daily_bars (
        session_date DATE NOT NULL,
        open DOUBLE NOT NULL,
        high DOUBLE NOT NULL,
        low DOUBLE NOT NULL,
        close DOUBLE NOT NULL,
        volume DOUBLE NOT NULL,
        finality VARCHAR NOT NULL,
        adjustment_revision VARCHAR NOT NULL,
        provider VARCHAR NOT NULL,
        receive_ts TIMESTAMPTZ NOT NULL,
        PRIMARY KEY (session_date, adjustment_revision)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS live_observations (
        observation_id VARCHAR PRIMARY KEY,
        instrument_id VARCHAR NOT NULL,
        asset_class VARCHAR NOT NULL,
        price DOUBLE,
        volume DOUBLE,
        event_ts TIMESTAMPTZ,
        receive_ts TIMESTAMPTZ NOT NULL,
        provider VARCHAR NOT NULL,
        data_status VARCHAR NOT NULL,
        degraded BOOLEAN NOT NULL,
        is_canonical BOOLEAN NOT NULL,
        request_set_id VARCHAR,
        lkg_fallback BOOLEAN NOT NULL DEFAULT FALSE,
        metadata_json VARCHAR
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS market_coverage (
        provider VARCHAR NOT NULL,
        adjustment_revision VARCHAR NOT NULL,
        instrument_id VARCHAR NOT NULL,
        session DATE NOT NULL,
        request_set_id VARCHAR NOT NULL,
        status VARCHAR NOT NULL,
        updated_at TIMESTAMPTZ NOT NULL,
        PRIMARY KEY (provider, adjustment_revision, instrument_id, session, request_set_id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS provider_state (
        provider_key VARCHAR PRIMARY KEY,
        role VARCHAR NOT NULL,
        status VARCHAR NOT NULL,
        last_success_at TIMESTAMPTZ,
        last_error VARCHAR,
        updated_at TIMESTAMPTZ NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS live_watchlist (
        instrument_id VARCHAR PRIMARY KEY,
        reason VARCHAR NOT NULL,
        priority INTEGER NOT NULL,
        updated_at TIMESTAMPTZ NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS universe_members (
        preset VARCHAR NOT NULL,
        symbol VARCHAR NOT NULL,
        member_rank INTEGER NOT NULL,
        dollar_volume DOUBLE,
        source VARCHAR NOT NULL,
        updated_at TIMESTAMPTZ NOT NULL,
        PRIMARY KEY (preset, symbol)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS universe_state (
        preset VARCHAR PRIMARY KEY,
        status VARCHAR NOT NULL,
        n_members INTEGER NOT NULL,
        source VARCHAR NOT NULL,
        last_refresh_at TIMESTAMPTZ,
        last_error VARCHAR,
        updated_at TIMESTAMPTZ NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS user_watchlist (
        symbol VARCHAR PRIMARY KEY,
        note VARCHAR,
        created_at TIMESTAMPTZ NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS custom_universe (
        symbol VARCHAR PRIMARY KEY,
        created_at TIMESTAMPTZ NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS feature_rows (
        instrument_id VARCHAR NOT NULL,
        as_of_date DATE NOT NULL,
        horizon INTEGER NOT NULL,
        asset_class VARCHAR NOT NULL,
        features_json VARCHAR NOT NULL,
        live_overlay_json VARCHAR,
        PRIMARY KEY (instrument_id, as_of_date, horizon, asset_class)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS label_rows (
        instrument_id VARCHAR NOT NULL,
        as_of_date DATE NOT NULL,
        horizon INTEGER NOT NULL,
        asset_class VARCHAR NOT NULL,
        label DOUBLE,
        label_available_date DATE NOT NULL,
        target_start DATE NOT NULL,
        target_end DATE NOT NULL,
        target_version VARCHAR NOT NULL,
        matured BOOLEAN NOT NULL,
        PRIMARY KEY (instrument_id, as_of_date, horizon, asset_class)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS prediction_rows (
        prediction_id VARCHAR PRIMARY KEY,
        instrument_id VARCHAR NOT NULL,
        as_of_date DATE NOT NULL,
        horizon INTEGER NOT NULL,
        asset_class VARCHAR NOT NULL,
        family VARCHAR NOT NULL,
        value DOUBLE NOT NULL,
        decision_epoch_id VARCHAR,
        created_at TIMESTAMPTZ NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS allocation_snapshots (
        snapshot_id VARCHAR PRIMARY KEY,
        created_at TIMESTAMPTZ NOT NULL,
        formula_version VARCHAR NOT NULL,
        payload_json VARCHAR NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS sec_extracts (
        accession VARCHAR NOT NULL,
        provider_model VARCHAR NOT NULL,
        prompt_version VARCHAR NOT NULL,
        accepted_at TIMESTAMPTZ,
        payload_json VARCHAR NOT NULL,
        PRIMARY KEY (accession, provider_model, prompt_version)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS ensemble_state (
        stream VARCHAR PRIMARY KEY,
        weights_json VARCHAR NOT NULL,
        horizon_influence_json VARCHAR NOT NULL,
        version VARCHAR NOT NULL,
        updated_at TIMESTAMPTZ NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS recommendation_outcomes (
        outcome_id VARCHAR PRIMARY KEY,
        recommendation_id VARCHAR NOT NULL,
        tick_id VARCHAR NOT NULL,
        source VARCHAR NOT NULL,
        instrument_id VARCHAR NOT NULL,
        horizon INTEGER NOT NULL,
        realized_return DOUBLE,
        outcome_snapshot_id VARCHAR,
        scored_at TIMESTAMPTZ NOT NULL,
        UNIQUE (recommendation_id, horizon)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS override_score_rows (
        score_id VARCHAR PRIMARY KEY,
        outcome_snapshot_id VARCHAR,
        tick_id VARCHAR NOT NULL,
        instrument_id VARCHAR NOT NULL,
        payload_json VARCHAR NOT NULL,
        created_at TIMESTAMPTZ NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS llm_transcripts (
        transcript_id VARCHAR PRIMARY KEY,
        tick_id VARCHAR,
        kind VARCHAR NOT NULL,
        instrument_id VARCHAR,
        ticker VARCHAR,
        model VARCHAR,
        prompt_version VARCHAR,
        status VARCHAR,
        system_prompt VARCHAR,
        user_prompt VARCHAR,
        response_text VARCHAR,
        error VARCHAR,
        prompt_tokens INTEGER,
        completion_tokens INTEGER,
        total_tokens INTEGER,
        created_at TIMESTAMPTZ NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS research_evidence_packs (
        pack_id VARCHAR PRIMARY KEY,
        ticker VARCHAR NOT NULL,
        utc_date DATE NOT NULL,
        cache_key VARCHAR NOT NULL,
        instrument_id VARCHAR,
        tick_id VARCHAR,
        quant_action VARCHAR,
        accession VARCHAR,
        material_change BOOLEAN,
        web_search_used BOOLEAN,
        payload_json VARCHAR NOT NULL,
        created_at TIMESTAMPTZ NOT NULL,
        UNIQUE (ticker, cache_key)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS llm_cost_ledger (
        entry_id VARCHAR PRIMARY KEY,
        utc_date DATE NOT NULL,
        provider VARCHAR NOT NULL,
        model VARCHAR NOT NULL,
        kind VARCHAR NOT NULL,
        ticker VARCHAR,
        prompt_tokens INTEGER,
        completion_tokens INTEGER,
        estimated_usd DOUBLE NOT NULL,
        created_at TIMESTAMPTZ NOT NULL
    )
    """,
)


FOUNDATION_TABLES: tuple[str, ...] = (
    "schema_meta",
    "writer_lease",
    "job_lease",
    "issuers",
    "instruments",
    "instrument_aliases",
    "decision_epochs",
    "online_update_ledger",
    "feature_snapshots",
    "outcome_snapshots",
    "request_sets",
    "ticks",
    "tick_observations",
    "tick_predictions",
    "tick_recommendations",
    "watermarks",
    "alert_outbox",
    "portfolio_lots",
    "portfolio_meta",
    "btc_sleeve_state",
    "runtime_events",
    "data_health_snapshots",
    "model_change_journal",
    "equity_daily_bars",
    "btc_daily_bars",
    "live_observations",
    "market_coverage",
    "provider_state",
    "live_watchlist",
    "universe_members",
    "universe_state",
    "user_watchlist",
    "custom_universe",
    "feature_rows",
    "label_rows",
    "prediction_rows",
    "allocation_snapshots",
    "sec_extracts",
    "ensemble_state",
    "recommendation_outcomes",
    "override_score_rows",
    "llm_transcripts",
    "research_evidence_packs",
    "llm_cost_ledger",
)

ALTER_STATEMENTS: tuple[str, ...] = (
    "ALTER TABLE llm_transcripts ADD COLUMN IF NOT EXISTS prompt_tokens INTEGER",
    "ALTER TABLE llm_transcripts ADD COLUMN IF NOT EXISTS completion_tokens INTEGER",
    "ALTER TABLE llm_transcripts ADD COLUMN IF NOT EXISTS total_tokens INTEGER",
)
