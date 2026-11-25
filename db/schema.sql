-- ============================================================================
-- EQUORA - Ürün & Girdi/Çıktı (Stok Hareket) Veritabanı Şeması
-- ============================================================================

CREATE TABLE IF NOT EXISTS products (
    product_id      SERIAL PRIMARY KEY,
    barcode         VARCHAR(32) UNIQUE NOT NULL,
    product_name    VARCHAR(150) NOT NULL,
    category        VARCHAR(64) NOT NULL,
    unit_price      NUMERIC(10, 2) NOT NULL CHECK (unit_price >= 0),
    shelf_life_days INT NOT NULL CHECK (shelf_life_days > 0)
);

CREATE TABLE IF NOT EXISTS stock_movements (
    movement_id     BIGSERIAL PRIMARY KEY,
    store_id        INT NOT NULL DEFAULT 1,
    product_id      INT NOT NULL REFERENCES products(product_id) ON DELETE CASCADE,
    expiry_date     DATE NOT NULL,
    movement_type   VARCHAR(10) NOT NULL CHECK (movement_type IN ('IN', 'OUT', 'WASTE')),
    quantity        INT NOT NULL CHECK (quantity > 0),
    movement_date   TIMESTAMPTZ NOT NULL,
    discount_rate   NUMERIC(5, 4) NOT NULL DEFAULT 0.0000 CHECK (discount_rate >= 0 AND discount_rate <= 1)
);

CREATE INDEX IF NOT EXISTS idx_movements_prod_exp_date
    ON stock_movements (store_id, product_id, expiry_date, movement_date);

CREATE OR REPLACE VIEW v_stock_summary AS
SELECT
    m.store_id,
    m.product_id,
    p.barcode,
    p.product_name,
    p.category,
    m.expiry_date,
    MIN(m.movement_date) FILTER (WHERE m.movement_type = 'IN') AS first_in_date,
    COALESCE(SUM(m.quantity) FILTER (WHERE m.movement_type = 'IN'), 0) AS total_in,
    COALESCE(SUM(m.quantity) FILTER (WHERE m.movement_type = 'OUT'), 0) AS total_out,
    COALESCE(SUM(m.quantity) FILTER (WHERE m.movement_type = 'WASTE'), 0) AS total_waste,
    COALESCE(SUM(m.quantity) FILTER (WHERE m.movement_type = 'IN'), 0)
      - COALESCE(SUM(m.quantity) FILTER (WHERE m.movement_type IN ('OUT', 'WASTE')), 0) AS remaining_qty
FROM stock_movements m
JOIN products p ON m.product_id = p.product_id
GROUP BY m.store_id, m.product_id, p.barcode, p.product_name, p.category, m.expiry_date;
