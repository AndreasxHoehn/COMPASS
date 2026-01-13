# -------------------------------------------------
# test_debug_abs.r  –  robust R wrapper for ./compass
# -------------------------------------------------
library(jsonlite)

# ---------- Helper: turn a relative path into an absolute one ----------
abs_path <- function(p) normalizePath(p, winslash = "/", mustWork = FALSE)

# ---------- 1️⃣ Verify required input files ----------
inputs <- c(
  "data/BlockLand/artifical_cencus.csv",
  "data/BlockLand/artifical_survay.csv",
  "data/BlockLand/artificial_groups.csv"
)

missing <- inputs[!file.exists(inputs)]
if (length(missing) > 0) {
  stop("Missing input files (relative to the script location):\n",
       paste(missing, collapse = "\n"))
}

# ---------- 2️⃣ Ensure output directory exists ----------
out_dir <- dirname(abs_path("results/artificial_synthetic_population.csv"))
if (!dir.exists(out_dir)) {
  dir.create(out_dir, recursive = TRUE)
}
if (!file.access(out_dir, mode = 2) == 0) {
  stop("Cannot write to output directory: ", out_dir)
}

# ---------- 3️⃣ Build the payload with absolute paths ----------
payload <- list(
  constraints      = abs_path("data/BlockLand/artifical_cencus.csv"),
  microdata        = abs_path("data/BlockLand/artifical_survay.csv"),
  groups           = abs_path("data/BlockLand/artificial_groups.csv"),
  output           = abs_path("results/artificial_synthetic_population.csv"),
  validate         = abs_path("results/artificial_synthPopSurvey.csv"),
  initialTemp      = 1000.0,
  minTemp          = 0.001,
  coolingRate      = 0.997,
  reheatFactor     = 0.3,
  fitnessThreshold = 0.01,
  minImprovement   = 0.001,
  maxIterations    = 500000,
  windowSize       = 5000,
  change           = 100000,
  distance         = "NORM_EUCLIDEAN",
  useRandomSeed    = FALSE,          # boolean, not a string
  randomSeed       = 42
)

json_in <- toJSON(payload, auto_unbox = TRUE, digits = 15)

binary <- "./compass"

# ---------- 4️⃣ Run the binary, capture both stdout and stderr ----------
out <- system2(
  command = binary,
  input   = json_in,
  stdout  = TRUE,
  stderr  = TRUE,
  wait    = TRUE
)

rc <- attr(out, "status")   # NULL = success, otherwise numeric

if (!is.null(rc) && rc != 0) {
  cat("\n--- COMPASS STDERR START ---\n")
  cat(attr(out, "stderr"))
  cat("\n--- COMPASS STDERR END   ---\n")
  stop(sprintf("Compass exited with status %d", rc))
}

# ---------- 5️⃣ Parse and pretty‑print the JSON result ----------
result <- tryCatch(
  fromJSON(out),
  error = function(e) {
    stop("Failed to parse JSON from Compass output: ", e$message)
  }
)

cat("\n=== Compass response ===\n")
cat(toJSON(result, pretty = TRUE, auto_unbox = TRUE), "\n")