WITH bounds AS (
  SELECT date_trunc('day', timezone('Europe/Samara', now())) AS local_start
)
SELECT key, value::text
FROM (
  SELECT 'conversations_today' AS key, count(*) AS value
    FROM core_conversation c, bounds
    WHERE timezone('Europe/Samara', c.started_at) >= bounds.local_start
  UNION ALL
  SELECT 'messages_today', count(*)
    FROM core_message m, bounds
    WHERE timezone('Europe/Samara', m.created_at) >= bounds.local_start
  UNION ALL
  SELECT 'user_messages_today', count(*)
    FROM core_message m, bounds
    WHERE m.sender = 'user' AND timezone('Europe/Samara', m.created_at) >= bounds.local_start
  UNION ALL
  SELECT 'bot_messages_today', count(*)
    FROM core_message m, bounds
    WHERE m.sender = 'bot' AND timezone('Europe/Samara', m.created_at) >= bounds.local_start
  UNION ALL
  SELECT 'operator_messages_today', count(*)
    FROM core_message m, bounds
    WHERE m.sender = 'operator' AND timezone('Europe/Samara', m.created_at) >= bounds.local_start
  UNION ALL
  SELECT 'new_clients_today', count(*)
    FROM core_channelidentity ci, bounds
    WHERE timezone('Europe/Samara', ci.created_at) >= bounds.local_start
  UNION ALL
  SELECT 'clients_with_new_conversations_today', count(DISTINCT c.channel_identity_id)
    FROM core_conversation c, bounds
    WHERE timezone('Europe/Samara', c.started_at) >= bounds.local_start
  UNION ALL
  SELECT 'unique_clients_today', count(DISTINCT ci.external_id)
    FROM core_message m
    JOIN core_conversation c ON c.id = m.conversation_id
    JOIN core_channelidentity ci ON ci.id = c.channel_identity_id,
    bounds
    WHERE m.sender = 'user' AND timezone('Europe/Samara', m.created_at) >= bounds.local_start
  UNION ALL
  SELECT 'waiting_now', count(*) FROM core_conversation WHERE status = 'waiting_for_operator'
  UNION ALL
  SELECT 'operator_now', count(*) FROM core_conversation WHERE status = 'operator'
  UNION ALL
  SELECT 'bot_now', count(*) FROM core_conversation WHERE status = 'bot'
  UNION ALL
  SELECT 'closed_today', count(*)
    FROM core_conversation c, bounds
    WHERE c.status = 'closed'
      AND c.closed_at IS NOT NULL
      AND timezone('Europe/Samara', c.closed_at) >= bounds.local_start
) s
ORDER BY key;
