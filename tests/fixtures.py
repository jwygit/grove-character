def bundle():
    return {'id': 'ep1', 'source': 'recording', 'time_origin': 86400., 'duration': 10.,
            'faces': [{'id': 'ep1:f1', 'observation_id': 'ep1:f1', 'image': 'unused.jpg'}],
            'utterances': [{'id': 'u1', 'start': 1., 'end': 3., 'text': 'hello', 'voice_observation_id': 'ep1:v1'},
                           {'id': 'u2', 'start': 5., 'end': 5.4, 'text': 'yes'}],
            'observations': [dict(id='ep1:f1', modality='face', source='recording', start=86401., end=86401.,
                                  model='face-fixture', embedding=[1., 0.]),
                             dict(id='ep1:v1', modality='voice', source='recording', start=86401., end=86403.,
                                  model='voice-fixture', embedding=[1., 0.])]}


def predictions():
    return [{'utterance_id': 'u1', 'face_id': 'ep1:f1'}, {'utterance_id': 'u2', 'face_id': None}]


def episode():
    return {'episode_id': 'ep1', 'time_span': '1-00:00:00-00:00:10', 'start_seconds': 86400.,
            'end_seconds': 86410., 'day': 1, 'frame_captions': [
                {'frame_time_span': 'DAY1-00:00:00-00:00:10', 'caption': 'Two people are talking.',
                 'day': 1, 'start_seconds': 86400., 'end_seconds': 86410.,
                 'window_time_span': '1-00:00:00-00:00:10', 'window_description': 'Conversation'}]}
