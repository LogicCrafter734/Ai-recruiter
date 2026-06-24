def calculate_behavioral_multiplier(signals):
    if not signals:
        return 0.5

    score_components = []
    score_components.append(signals.get("profile_completeness_score", 50) / 100.0)
    score_components.append(1.0 if signals.get("open_to_work_flag") else 0.4)
    score_components.append(signals.get("recruiter_response_rate", 0.5))
    score_components.append(signals.get("interview_completion_rate", 0.5))
    
    offer_rate = signals.get("offer_acceptance_rate", -1)
    score_components.append(offer_rate if offer_rate >= 0 else 0.5)
    
    active_date_str = str(signals.get("last_active_date", "2026"))
    active_year = int(active_date_str.split("-")[0]) if "-" in active_date_str else 2026
    score_components.append(1.0 if active_year >= 2026 else 0.2)
    
    views = signals.get("profile_views_received_30d", 0)
    score_components.append(min(views / 30.0, 1.0))
    
    saves = signals.get("saved_by_recruiters_30d", 0)
    score_components.append(min(saves / 5.0, 1.0))
    
    github = signals.get("github_activity_score", -1)
    score_components.append(github / 100.0 if github >= 0 else 0.2)
    
    verifications = 0
    if signals.get("verified_email"): verifications += 0.5
    if signals.get("verified_phone"): verifications += 0.5
    score_components.append(verifications)
    
    return sum(score_components) / len(score_components)