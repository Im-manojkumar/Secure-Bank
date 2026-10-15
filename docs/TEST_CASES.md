# SecureBank test cases

All of these run automatically with `pytest`, locally and on every push and pull request through GitHub Actions (`.github/workflows/tests.yml`). The tests use generated data and an in-memory database, so they run without the Kaggle file. Each row names the test that checks it.

## 1. Machine learning (`tests/test_ml.py`)

| ID | Case | Expected result | Test |
|---|---|---|---|
| ML-01 | Dataset missing a column | Rejected with the column named | `test_load_data_rejects_missing_columns` |
| ML-02 | Dataset file not found | Error says where to put `creditcard.csv` | `test_load_data_missing_file_has_helpful_message` |
| ML-03 | Dataset with duplicate rows | Duplicates removed | `test_load_data_drops_duplicates` |
| ML-04 | Train, validation and test split by time | Every training row comes before every validation row, and those before every test row; 60/20/20 | `test_split_follows_time` |
| ML-04a | Rows with the same time at a split edge | They stay in the same set | `test_time_split_keeps_rows_with_the_same_time_together` |
| ML-04b | A period with no fraud | Refused with the period named, since models cannot be compared on it | `test_time_split_refuses_a_period_without_fraud` |
| ML-04c | Sparkov's separate test file | Validation is the latest 25% of the training file; the test file is used as it is | `test_sparkov_validation_is_the_latest_part_of_the_training_file` |
| ML-04d | Random split (`--random-split`, for comparison) | Splits don't overlap and each keeps the fraud rate | `test_random_split_is_stratified_and_disjoint` |
| ML-05 | Threshold choice | Threshold separates fraud from non-fraud when the scores allow it | `test_best_f1_threshold_separates_perfect_scores` |
| ML-05a | Cost-based threshold | Flags down to a large fraud when reviews are cheap; flags nothing when reviews cost more than any fraud | `test_cost_threshold_flags_fraud_worth_more_than_its_review` |
| ML-05b | Tied scores at the cost cut-off | Tied rows are flagged together | `test_cost_threshold_never_splits_tied_scores` |
| ML-05c | Cost of a threshold | Missed fraud amount plus review cost per alert | `test_alert_costs_add_missed_fraud_and_reviews` |
| ML-05d | Training with `--threshold-rule f1` | Rule and review cost saved; the other rule's result is reported; an unknown rule is refused | `test_train_records_the_threshold_rule_and_its_cost` |
| ML-06 | Full training run | Four models compared, model and reports saved, best model finds over 70% of fraud in the test data, split recorded as "time" | `test_train_saves_model_that_scores_new_rows` |

## 2. Login and roles (`tests/test_api.py`, `tests/test_pages.py`, `tests/test_security.py`)

| ID | Case | Expected result | Test |
|---|---|---|---|
| AU-01 | API call without logging in | HTTP 401, "Please log in first." | `test_endpoints_require_login` |
| AU-02 | Page visit without logging in | Redirect to the login page, which returns to the page afterwards | `test_pages_redirect_to_login_when_logged_out` |
| AU-03 | Wrong password vs unknown email | Same error for both, so emails can't be probed | `test_login_rejects_wrong_password_with_generic_message` |
| AU-04 | Log in, then log out | Session works, then ends | `test_login_me_and_logout` |
| AU-05 | Analyst tries to import | HTTP 403 | `test_analyst_cannot_import` |
| AU-06 | Analyst opens the import or audit page | "No access" page (403), and the link is hidden | `test_import_page_is_admin_only`, `test_audit_page_is_admin_only` |
| AU-07 | 5 wrong passwords in 15 minutes | 6th attempt refused with 429, even with the right password, and other accounts are unaffected | `test_login_blocked_after_repeated_failures` |
| AU-08 | Failures, then a success, then failures again | Count resets after the successful login | `test_successful_login_resets_the_failure_count` |
| AU-09 | `?next=https://evil.example` on login | Ignored, user goes to the dashboard | `test_login_page_ignores_offsite_next` |

## 3. Importing transactions (`tests/test_api.py`)

| ID | Case | Expected result | Test |
|---|---|---|---|
| IM-01 | Valid CSV of 1,000 rows | All stored, each scored, and an Open case for every flagged row | `test_import_stores_transactions_predictions_and_cases` |
| IM-02 | Same file imported twice | Second import adds nothing | `test_reimporting_same_file_skips_duplicates` |
| IM-03 | Rows with text in a number column, a negative amount, or `Class` = 7 | Those rows skipped and counted, and the rest imported | `test_import_skips_invalid_rows_and_reports_them` |
| IM-04 | File missing a column | Rejected (400), and the failed import is still recorded | `test_import_rejects_missing_columns_and_records_failure` |
| IM-05 | Non-CSV file, or no file | Rejected (400) | `test_import_rejects_non_csv_and_missing_file` |
| IM-06 | File without `Class` | Imported, with no dataset label | `test_import_works_without_class_column` |
| IM-07 | Same rows, `Time` written as `10` in one file and `10.0` in another | Treated as duplicates | `test_duplicate_check_ignores_number_formatting` |

## 4. Search, prediction and cases (`tests/test_api.py`)

| ID | Case | Expected result | Test |
|---|---|---|---|
| TX-01 | List transactions | Paginated, at most 100 per page | `test_list_transactions_paginates` |
| TX-02 | Filter by prediction, amount, date and case status | Only matching rows returned | `test_list_transactions_filters` |
| TX-03 | Filter with bad values | 400 with a message | `test_list_transactions_rejects_bad_filters` |
| TX-04 | Open one transaction, or a missing one | All 30 model inputs shown, or 404 | `test_get_transaction_includes_features` |
| PR-01 | Check a valid transaction | Result, risk score and model version returned | `test_predict_scores_a_transaction` |
| PR-02 | Check with a missing or non-numeric field | 400 listing each problem | `test_predict_validates_input` |
| CA-01 | Alerts list | Highest risk first, and an unknown status rejected | `test_alerts_sorted_by_risk` |
| CA-02 | Review a case: Under Review, then Resolved, then reopened | Reviewer recorded. Resolving needs an outcome. The prediction is unchanged. Reopening clears the outcome. | `test_case_review_flow` |
| CA-03 | Bad case updates | Unknown status, unknown field, empty body, outcome without resolving, or non-text notes give 400. A missing case gives 404. | `test_case_update_validation` |
| DB-01 | Dashboard summary | Totals match, and the hourly trend adds up | `test_dashboard_summary` |
| DB-02 | Dashboard with no data | Zeros, no errors | `test_dashboard_summary_empty_database` |

## 5. Audit log (`tests/test_security.py`)

| ID | Case | Expected result | Test |
|---|---|---|---|
| AL-01 | Failed login, login, logout | Three entries, with email and IP address | `test_logins_and_logout_are_audited` |
| AL-02 | Case resolved | Entry records the status and outcome before and after | `test_case_update_is_audited_with_before_and_after` |
| AL-03 | Rejected case update | Nothing logged | `test_rejected_case_update_is_not_audited` |
| AL-04 | Import that works and one that fails | Both logged with the file name | `test_imports_are_audited` |
| AL-05 | User created from the command line | Logged with the role | `test_cli_create_user_is_audited` |
| AL-06 | Audit log API | Admin only, newest first, and filters by action and subject | `test_audit_api_is_admin_only_and_filters` |

## 6. Web security (`tests/test_security.py`, `tests/test_pages.py`)

| ID | Case | Expected result | Test |
|---|---|---|---|
| WS-01 | POST from another website | 403 "Cross-site request blocked." | `test_cross_site_api_writes_are_blocked` |
| WS-02 | POST from SecureBank's own pages | Allowed | `test_same_site_api_writes_are_allowed` |
| WS-03 | Security headers | CSP, `nosniff`, `DENY` framing, and `no-store` on the API | `test_security_headers` |
| WS-04 | Every page | No inline scripts or styles that the CSP would block | `test_pages_have_no_inline_scripts_or_styles` |
| WS-05 | Unknown page or API route | HTML 404 page, or JSON error | `test_unknown_page_shows_html_404`, `test_api_errors_are_json` |

## 7. More datasets and explanations (`tests/test_datasets.py`)

| ID | Case | Expected result | Test |
|---|---|---|---|
| DS-01 | Kaggle, Sparkov and PaySim files | Each format recognised from its columns | `test_formats_are_recognised_from_their_columns` |
| DS-02 | Sparkov rows with a negative amount, unknown category, bad date or label 3 | Those rows dropped; names, street, job, ZIP, gender and full card number never kept; card reduced to its last 4 digits | `test_sparkov_cleaning_drops_bad_rows_and_personal_data` |
| DS-03 | Sparkov model inputs | Hour 0 to 23, positive age, distance of 0 or more, exactly one merchant category per row | `test_sparkov_features_are_readable` |
| DS-04 | PaySim transfer that empties the sender's account | Inputs mark the emptied account, the receiver mismatch and the hour | `test_paysim_features_include_balance_mismatches` |
| DS-05 | Training per format | Each format has its own model, with a typical genuine row saved for explanations | `test_each_format_trains_its_own_model` |
| DS-06 | Explaining flagged fake Sparkov fraud (large late-night online buys) | Amount, time of day or category lead the reasons; at most 3 reasons, each with value, typical value and effect | `test_explanations_name_the_planted_pattern` |
| DS-07 | Import a Sparkov file and a PaySim file | Scored by the matching model, right currency (USD, none), reasons stored for suspicious rows, card number never stored, re-import adds nothing | `test_import_scores_each_format_with_its_model` |
| DS-08 | Low-risk rows | No explanation stored | `test_low_risk_rows_store_no_explanation` |
| DS-09 | File matching no format, or Sparkov missing `dob` | 400; the message names the closest format and the missing column | `test_import_rejects_unknown_format`, `test_import_names_missing_columns_of_the_closest_format` |
| DS-10 | Import when the format has no trained model | 503 naming the training command | `test_import_without_a_trained_model_is_refused` |
| DS-11 | `GET /api/models` | All three formats, their form fields and options, and each trained model's threshold | `test_models_endpoint_lists_formats_and_models` |
| DS-12 | Check a Sparkov transaction | Result plus the reasons for its score | `test_predict_with_a_dataset_explains_the_score` |
| DS-13 | Check with bad values in each format | 400 listing each problem; unknown format refused | `test_predict_validates_each_format`, `test_transaction_filter_rejects_unknown_dataset` |
| DS-14 | Dashboard with euro and dollar imports | Flagged amounts totalled per currency, never mixed | `test_dashboard_totals_flagged_amounts_per_currency` |
| DS-15 | Dashboard and alerts limited to one format, by day or by hour | Totals, cases, trend and alerts cover only that format; an unknown format gives 400 | `test_dashboard_and_alerts_filter_by_dataset` |
| DS-16 | Card history inputs | Count only the same card's earlier payments: within an hour, within a day, amount against its average, hours since its last | `test_card_history_counts_only_earlier_payments_of_the_same_card` |
| DS-17 | Card history with past payments that are not scored | Earlier payments of the same card count; other cards and later payments do not | `test_card_history_uses_earlier_payments_that_are_not_scored` |
| DS-18 | Card IDs | The same number always gives the same 16-character ID, which does not contain the number | `test_card_ids_are_stable_and_hide_the_number` |
| DS-19 | Import a Sparkov file in two parts | Every payment stored with its card ID; the second part finds the first part's payments as history | `test_import_compares_payments_with_the_card_history_already_stored` |
| DS-20 | `init-db` on a database from before `card_id` existed | Column and index added; running again changes nothing | `test_init_db_adds_columns_missing_from_an_older_database` |

## 7a. Learning from analysts (`tests/test_feedback.py`)

| ID | Case | Expected result | Test |
|---|---|---|---|
| FB-01 | Cases resolved as Confirmed fraud and Legitimate | Returned as rows labelled 1 and 0 with their inputs; other formats not mixed in | `test_resolved_cases_become_labelled_rows` |
| FB-02 | Reviewed rows that include test rows | Test rows dropped and counted; the rest added to training only | `test_reviewed_rows_join_training_but_never_validation_or_test` |
| FB-03 | `flask retrain` | Old model copied to `previous/`, new model records its reviewed cases, retrain audited | `test_retrain_command_keeps_the_old_model_and_logs_it` |
| FB-04 | `flask retrain` with an unknown format | Refused with a clear message | `test_retrain_rejects_an_unknown_format` |

## 7b. Drift check (`tests/test_drift.py`)

| ID | Case | Expected result | Test |
|---|---|---|---|
| DR-01 | New rows spread like the training data, then amounts 20 times larger | Stable with no message; then a major shift led by Amount, with a warning | `test_psi_is_near_zero_for_the_same_spread_and_large_for_a_shift` |
| DR-02 | Fewer than 200 rows | Not judged | `test_drift_is_not_judged_on_a_handful_of_rows` |
| DR-03 | Import a normal file, then a shifted one | First stable; second marked major in its message and on the dashboard, only for its own format | `test_drift_shows_on_the_import_and_the_dashboard` |

## 7c. Live scoring API (`tests/test_live.py`)

| ID | Case | Expected result | Test |
|---|---|---|---|
| LV-01 | No key, a wrong key, a valid key, a revoked key | 401, 401, 200, 401 | `test_score_needs_a_valid_active_key` |
| LV-02 | A risky Sparkov payment with a card number | Review or decline with reasons; saved under the key's source with a case; card number not stored | `test_live_score_decides_stores_and_opens_a_case` |
| LV-03 | The same call sent twice | Second answer marked duplicate, same transaction, nothing saved twice | `test_a_retried_call_is_answered_without_storing_twice` |
| LV-04 | Three payments on one card within minutes | All linked by card ID; a fourth sees three in the past hour | `test_payments_on_the_same_card_build_its_history` |
| LV-05 | Score at or above `DECLINE_SCORE` | Declined | `test_very_high_scores_are_declined` |
| LV-06 | Unknown format, negative amount, card number with dashes | 400 with the problems listed | `test_bad_requests_are_explained` |
| LV-07 | `create-api-key`, then `revoke-api-key` | Key printed once, only its hash stored, revoked; both audited | `test_api_keys_are_made_and_revoked_from_the_command_line` |

## 7d. Two-factor login (`tests/test_two_factor.py`)

| ID | Case | Expected result | Test |
|---|---|---|---|
| TF-01 | Codes for RFC 6238's test secret | Match the RFC's published codes | `test_codes_match_the_rfc_test_vectors` |
| TF-02 | A code used twice, one step late, three steps late, 5 digits | Accepted once; refused again; accepted; refused; refused | `test_a_code_works_once_within_its_window` |
| TF-03 | Setup with a wrong code | Refused; two-factor stays off | `test_setup_needs_a_correct_code` |
| TF-04 | Log in with two-factor on: no code, wrong code, right code | Asked for the code without logging in; failed login audited; logged in | `test_login_asks_for_the_code_once_it_is_on` |
| TF-05 | Admin without two-factor when it is required | API 403 and pages redirect to Account until set up; cannot turn it off afterwards | `test_admins_must_set_it_up_before_anything_else` |
| TF-06 | Analyst turns it off | Needs the right password and a code; both changes audited | `test_analyst_can_turn_it_off_with_password_and_code` |

## 7e. Production deployment (`tests/test_deployment.py`)

| ID | Case | Expected result | Test |
|---|---|---|---|
| DP-01 | Production mode without `SECRET_KEY` | Refuses to start, naming the missing setting | `test_production_refuses_to_start_without_a_secret_key` |
| DP-02 | Production mode with the development `CARD_ID_KEY` | Refuses to start | `test_production_refuses_the_development_card_key` |
| DP-03 | Production mode with a short key, or the example value from `deploy/env.example` | Refuses to start | `test_production_refuses_weak_or_example_secrets` |
| DP-04 | Log in in production mode, even with `SESSION_COOKIE_SECURE` off | Cookie marked `Secure`; `Strict-Transport-Security` sent | `test_production_forces_secure_cookies_and_hsts` |
| DP-05 | Development mode | No `Strict-Transport-Security` header | `test_development_sends_no_hsts` |
| DP-06 | Failed login through a trusted proxy with `X-Forwarded-For: 203.0.113.7` | Audit log records 203.0.113.7 | `test_behind_a_proxy_the_visitor_address_is_logged` |
| DP-07 | The same header without `TRUST_PROXY` | Ignored, so it cannot dodge the login limit | `test_forwarded_address_is_ignored_without_a_proxy` |
| DP-08 | `GET /api/health` | 200 with the database ok; 503 when the database fails | `test_health_check_reports_the_database`, `test_health_check_fails_when_the_database_is_down` |
| DP-09 | An unexpected error in a page | 500 page without the error's details; the traceback is logged | `test_unexpected_errors_are_logged_and_hidden` |
| DP-10 | `flask backup-db --keep 2` with three older backups | A readable copy of the database; only the two newest backups kept | `test_backup_copies_the_database_and_keeps_the_newest` |
| DP-11 | `flask backup-db` on an in-memory database | Refused with a message | `test_backup_needs_a_database_file` |

## 8. Saved reports and CSV downloads (`tests/test_reports.py`)

| ID | Case | Expected result | Test |
|---|---|---|---|
| RP-01 | Save "suspicious Sparkov" as a report | Unknown and empty filters dropped; totals, amounts, currency and open cases match the same search | `test_saved_report_totals_match_the_search` |
| RP-02 | Resolve a case as confirmed fraud, then open the report | The report shows the new status and outcome, because it is run again each time | `test_report_reflects_case_reviews` |
| RP-03 | Report with no filters | Covers every transaction, in every format | `test_report_with_no_filters_covers_everything` |
| RP-04 | Blank or 121-character name, bad amount, unknown case status, filters not an object | 400 with a message | `test_report_validation` |
| RP-05 | Two reports named "Weekly" and "weekly" | Second one refused with 409 | `test_report_names_are_unique` |
| RP-06 | Analyst deletes an admin's report, then their own; admin deletes theirs | 403, then 204, then 204 | `test_only_the_owner_or_an_admin_can_delete` |
| RP-07 | Download a report | CSV named after the report, one row per match, with dataset, currency, case status and top reason, and no personal data | `test_report_downloads_as_csv` |
| RP-08 | Download the current transaction search | Only matching rows; an unknown format gives 400 | `test_search_downloads_as_csv` |
| RP-09 | Import a file named `=HYPERLINK(evil).csv`, then download | The cell starts with `'`, so spreadsheets show it as text instead of running it | `test_csv_cells_cannot_run_as_formulas` |
| RP-10 | Download more rows than `EXPORT_MAX_ROWS` | Refused with the row count; saving, deleting and each real download are in the audit log, but refused downloads are not | `test_downloads_are_limited_and_audited` |
| RP-11 | Reports and downloads without logging in | 401 | `test_reports_need_a_login` |
| RP-12 | Database file in use | Runs in write-ahead-log mode, so long reads don't block logins | `test_file_databases_use_write_ahead_logging` |

## 9. Manual checks in a browser


These were run in Chromium against the full Kaggle dataset (283,726 transactions).

| ID | Steps | Expected result |
|---|---|---|
| UI-01 | Log in with a wrong password, then the right one | Error message, then the dashboard |
| UI-02 | Hover a bar on either dashboard chart | Tooltip with the hour and count |
| UI-03 | Search transactions with Prediction set to Suspicious and Min amount 500 | 27 results, all suspicious and €500 or more |
| UI-04 | Open an alert, choose Resolved without an outcome, then save | "Choose an outcome before resolving the case." |
| UI-05 | Choose an outcome, type notes containing `<script>`, then save | "Review saved." The notes show as plain text and nothing runs. |
| UI-06 | Check a transaction, starting from transaction 542 | Values filled in, then "Low risk" with its score and the threshold marker |
| UI-07 | As an admin, upload a CSV on the Import page | Progress bar, then the result message and a new history row |
| UI-08 | Open the Audit log as an admin | Every action above listed, and case entries link to the case |
| UI-09 | Repeat UI-01 with the system set to dark mode | Readable dark theme |
| UI-10 | Use a 390 px wide phone screen | No sideways scrolling, and the menu scrolls on one line |
| UI-11 | Watch the browser console on every page | No script errors and no Content-Security-Policy violations |
| UI-12 | Search Sparkov, Suspicious, Min amount 500, save it as "Large Sparkov alerts", then open it from Reports (1.1 million transactions imported) | 1,156 transactions, all suspicious, $1,079,813.90; opens in well under a second |
| UI-13 | On that report, click Download CSV, then Open in Transactions | `securebank-large-sparkov-alerts.csv` with 1,156 rows; the Transactions page opens with the same filters filled in |
| UI-14 | Click Download CSV on Transactions with no filters | "This matches 1,139,445 transactions. Narrow the filters…", and no file is saved |

Two-factor login was checked in Chromium: an admin without it lands on the Account page and is sent back there from other pages; after adding the key to an app and entering a code, the dashboard opens; on the next login the page asks for the code and then logs in.

The Docker setup was checked by building the image and running it behind Caddy: the app refused to start without its keys, then reported healthy as a non-root user; login over HTTPS set a `Secure` cookie with `Strict-Transport-Security`; the audit log recorded the visitor's address rather than Caddy's; and `flask backup-db` wrote a copy to the data volume.
