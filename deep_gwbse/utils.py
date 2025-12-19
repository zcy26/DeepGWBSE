import os 
import json
import subprocess

def jobdone(task_dir: str) -> bool:
    task = os.path.basename(task_dir)
    if task == '01-density':
        result = subprocess.run(['grep', 'DONE', os.path.join(task_dir, 'scf.out')], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return result.returncode == 0
    elif task == "02-wfn":
        res1 = subprocess.run(['grep', 'DONE', os.path.join(task_dir, 'wfn.out')], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        res2 = subprocess.run(['grep', 'TOTAL', os.path.join(task_dir, 'parabands.out')], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        res3 = subprocess.run(['grep', 'DONE', os.path.join(task_dir, 'wfn.pp.out')], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return res1.returncode == 0 and res2.returncode == 0 and res3.returncode == 0
    elif task == "03-wfnq":
        res1 = subprocess.run(['grep', 'DONE', os.path.join(task_dir, 'wfn.out')], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        res2 = subprocess.run(['grep', 'DONE', os.path.join(task_dir, 'wfn.pp.out')], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        res3 = subprocess.run(['grep', 'alpha', os.path.join(task_dir, 'pseudo.out')], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return res1.returncode == 0 and res2.returncode == 0 and res3.returncode == 0
    elif task == '05-band':
        res1 = subprocess.run(['grep', 'DONE', os.path.join(task_dir, 'wfn.out')], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        res2 = subprocess.run(['grep', 'DONE', os.path.join(task_dir, 'wfn.pp.out')], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        res3 = subprocess.run(['grep', 'DONE', os.path.join(task_dir, 'bands.out')], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return res1.returncode == 0 and res2.returncode == 0 and res3.returncode == 0
    elif task == '06-wfnq-nns':
        res1 = subprocess.run(['grep', 'DONE', os.path.join(task_dir, 'wfn.out')], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        res2 = subprocess.run(['grep', 'DONE', os.path.join(task_dir, 'wfn.pp.out')], stdout=subprocess.PIPE, stderr=subprocess.PIPE)     
        return res1.returncode == 0 and res2.returncode == 0   
    elif task == "11-epsilon":
        result = subprocess.run(['grep', 'Job Done', os.path.join(task_dir, 'epsilon.out')], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return result.returncode == 0
    elif task == "12-epsilon-nns":
        result = subprocess.run(['grep', 'Job Done', os.path.join(task_dir, 'epsilon.out')], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return result.returncode == 0        
    elif task == "13-sigma":
        result = subprocess.run(['grep', 'Job Done', os.path.join(task_dir, 'sigma.out')], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return result.returncode == 0
    elif task == "14-inteqp":
        # grep 'Job Done' inteqp.log
        result = subprocess.run(['grep', 'Job Done', os.path.join(task_dir, 'inteqp.log')], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return result.returncode == 0
    elif task == "17-wfn_fi":
        res1 = subprocess.run(['grep', 'DONE', os.path.join(task_dir, 'wfn.out')], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        res2 = subprocess.run(['grep', 'DONE', os.path.join(task_dir, 'wfn.pp.out')], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return res1.returncode == 0 and res2.returncode == 0
    elif task == "18-kernel":
        # grep 'TOTAL' kernel.out
        result = subprocess.run(['grep', 'TOTAL', os.path.join(task_dir, 'kernel.out')], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return result.returncode == 0
    elif task == '19-absorption':
        # grep 'TOTAL' absorption.out
        result = subprocess.run(['grep', 'TOTAL', os.path.join(task_dir, 'absorption.out')], stdout=subprocess.PIPE, stderr=subprocess.PIPE)    
        return result.returncode == 0
    else:
        return 'unknown task'


def check_flows_status(flows: str = './flows-semi', dump: bool = True):
    print(f"Checking flows status in {flows}...")
    flows_status = {}
    
    for root, dirs, _ in os.walk(flows):
        if '01-density' not in dirs:
            if "02-wfn" not in dirs:
                # gw augmentation
                if "17-wfn_fi" not in dirs:
                    # bse augmentation
                    continue
        
        flow_status = {"Yes": [], "No": [], "Unknown Job": []}
        for dir in filter(lambda d: d != 'pp', dirs):
            job_status = jobdone(os.path.join(root, dir))
            if job_status is True:
                flow_status["Yes"].append(dir)
            elif job_status is False:
                flow_status["No"].append(dir)
            else:
                flow_status["Unknown Job"].append(dir)
        
        # Sort once at the end for efficiency
        for key in flow_status:
            if flow_status[key]:  
                flow_status[key].sort()
        
        flows_status[root] = {k: ",".join(v) for k, v in flow_status.items()}
    
    if dump:
        output_file = f"{os.path.basename(flows)}_status.json"
        print(output_file, flows)
        with open(output_file, 'w') as f:
            json.dump(flows_status, f, indent=4, separators=(',', ': '))
        print(f"Flows status saved to {output_file}")
    
    return flows_status
