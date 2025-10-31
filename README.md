A/B Testing + CUPED + Retention Analytics + Churn Modeling (Real Dataset: Cookie Cats)

**Business Problem**

    Modern product teams run thousands of experiments annually.  
    This lab mimics a real product rollout decision:
    Should we launch Variant B or keep Variant A?

    Conversion lift
    Retention guardrails
    CUPED-adjusted significance
    User engagement features
    Churn risk


This project simulates a real-world product experimentation workflow like those used at Google, Meta, LinkedIn, and YouTube.
It demonstrates how to run rigorous tests, reduce variance, analyze retention, and model churn using both synthetic and real user data.

**Demo** - https://huggingface.co/spaces/aakash-malhan/growth-experimentation-lab-ab-testing-cuped-retention-churn

<img width="1685" height="373" alt="Screenshot 2025-10-31 140029" src="https://github.com/user-attachments/assets/b6ffdd0b-0b58-400c-9bae-f93a58855c83" />
<img width="1416" height="770" alt="Screenshot 2025-10-31 140055" src="https://github.com/user-attachments/assets/666e77cd-1b93-4d54-8272-39f6cda9ff12" />
<img width="1492" height="509" alt="Screenshot 2025-10-31 140120" src="https://github.com/user-attachments/assets/6c7bed09-fc6b-4c9e-8419-28f2729d660b" />
<img width="1676" height="224" alt="Screenshot 2025-10-31 143918" src="https://github.com/user-attachments/assets/c285f128-1d79-4792-a4bb-b3737846afba" />


**Key Capabilities**

    Experimentation      Two-sample tests, bootstrap CI, two-proportion z-test
    Variance Reduction   CUPED adjustment + variance reduction %
    Retention Analytics  Daily retention & funnel stages
    Churn Modeling       Logistic regression with class balance, threshold tuning
    Data Sources         Synthetic users + Cookie Cats real dataset
    Outputs              Product insights, lift estimates, PM-ready summary

**Example Results**

    Variant B improved conversion by ~+2.27pp (significant)
    CTR up ~5pp
    No retention drop
    Churn model AUC ~0.80–0.90 (realistic thresholding & balancing)
    Recommendation: Scale Variant B while monitoring guardrails (latency, retention dips, complaints).

