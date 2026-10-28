"""Immutable, scoped model learning; candidates are NOT validated defaults.

Geometry belongs to a fixture, constitutive laws to a material/cell family,
and numerical methods to a solver revision. Never transfer one kind as another.
Compatible development seeds carry forward even after a core update, but
their scores/checkpoints do not: recheck is explicit and mandatory.
"""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path

from fsri30_data import save,sha

ROOT=Path(__file__).resolve().parents[1]
REGISTRY=ROOT/'data/model-registry'
KINDS={'geometry_prior','constitutive_law','numerical_method'}
STATES={'candidate','numerically_verified','validated'}


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def record(kind,scope,parameters,evidence,*,compatibility,qualification='candidate',
           producer_core=None,registry=REGISTRY,activate_development=False):
    if kind not in KINDS or qualification not in STATES or not scope or not compatibility:
        raise ValueError('Evidence needs a kind, explicit scope, compatibility and qualification')
    if qualification!='candidate':
        required='numerical_checks' if qualification=='numerically_verified' else 'independent_validation'
        checks=evidence.get(required,{})
        if not checks.get('accepted'):
            raise ValueError('Cannot promote a model without passing '+required)
        if evidence.get('conditioned_transport_diagnostic') or not evidence.get('full_warmup_validation',True):
            raise ValueError('Conditioned suffix is not a qualified full model')
    value=dict(format_version=1,kind=kind,scope=scope,parameters=parameters,evidence=evidence,
        compatibility=compatibility,qualification=qualification,producer_core=producer_core,
        transferable_as=kind,calibrated=qualification=='validated',validation_passed=qualification=='validated')
    ident=digest(value);path=registry/'records'/f'{ident}.json'
    if path.exists():
        if json.loads(path.read_text())!=value:raise ValueError('Immutable evidence hash collision/change')
    else:save(path,value)
    if activate_development:
        import fcntl
        pointer=registry/'development-index.json'
        with (registry/'index.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            index=json.loads(pointer.read_text()) if pointer.exists() else {}
            index[kind+':'+scope]=dict(record=ident,updated_at=datetime.now(timezone.utc).isoformat())
            pending=pointer.with_suffix('.pending');save(pending,index);pending.replace(pointer)
    return ident


def seed(kind,scope,*,compatibility,consumer_core,registry=REGISTRY,record_id=None):
    # Resuming an existing search must use its ORIGINAL seed, even if that
    # search has since published a better development pointer.
    if record_id is None:
        pointer=registry/'development-index.json'
        if not pointer.exists():return None
        item=json.loads(pointer.read_text()).get(kind+':'+scope)
        if item is None:return None
        record_id=item['record']
    if len(record_id)!=64 or any(c not in '0123456789abcdef' for c in record_id):
        raise ValueError('Invalid immutable model record identifier')
    value=json.loads((registry/'records'/f'{record_id}.json').read_text())
    if digest(value)!=record_id:raise ValueError('Model evidence was modified')
    if value['kind']!=kind or value['scope']!=scope or value['compatibility']!=compatibility:
        raise ValueError('Model profile is not compatible with this material/geometry/stage')
    return dict(record=record_id,parameters=value['parameters'],qualification=value['qualification'],
        requires_recheck=value['producer_core']!=consumer_core or value['qualification']!='validated',
        reuse_scores=False,reuse_checkpoint=False,scope=scope)


def fsri30_compatibility(scene):
    # No numerical controls or fitted coefficients: this identifies the
    # fixture, materials, cell geometry and conditioned diagnostic stage.
    return dict(fixture='fsri30-reconstruction-v1',stage='conditioned-post-burst',
        geometry_sha256=digest(dict(cells=[dict(id=c['id'],geometry=c['geometry'],pose=c['pose'])
            for c in scene['cells']],shells=scene['shells'],
            positions=scene['metadata']['approximate_positions_m'])),
        dimensions='fsri30_suffix_model-v1')


def retain_fsri30_best(folder,*,registry=REGISTRY):
    best=json.loads((folder/'best.json').read_text())
    scene=json.loads((Path(best['trial'])/'scenario.json').read_text())
    result_path=Path(best['trial'])/'evaluation.json'
    if not result_path.exists():result_path=folder/'best.json'
    evidence=dict(result_path=str(result_path.resolve()),result_sha256=sha(result_path),
        trial=str(Path(best['trial']).resolve()),objective=best['objective'],
        reacted_components=best['cascade']['reacted_components'],
        conditioned_transport_diagnostic=True,full_warmup_validation=False,
        interpretation='Development geometry seed only, not a material law or accepted calibration')
    return record('geometry_prior','fsri30-conditioned-gas',best['parameters'],evidence,
        compatibility=fsri30_compatibility(scene),producer_core=best['native_sha256'],
        registry=registry,activate_development=True)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--retain-fsri30-best',type=Path,required=True)
    args=ap.parse_args();print(retain_fsri30_best(args.retain_fsri30_best))


if __name__=='__main__':main()
